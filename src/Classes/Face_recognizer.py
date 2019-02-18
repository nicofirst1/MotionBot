import glob
import math
import operator
import os
import random
import threading
import time
from collections import Counter
from threading import Thread

import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from face_recognition.face_recognition_cli import image_files_in_folder
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors, preprocessing, svm
from sklearn.decomposition import PCA
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ConversationHandler, CommandHandler, MessageHandler, CallbackQueryHandler, Filters
from tqdm import trange

from Path import Path as pt
from Utils.serialization import dump_pkl, load_pkl


class FaceRecognizer(Thread):
    """Class dedicated to face recognition
    Each face is saved in a folder inside faces_dir with the following syntax s_idx_subjectName, where idx is the
    number of direcoties inside faces_dir and subjectName is the name of the person the face belogns to.

    Attributes:
        faces_dir : the directory Faces
        unknown : the Unknown direcotry
        recognizer_path : the path to the recognizer object
        stop_event : The event to handle thread stopping

        recognizer: the classifier used for face rocognition
        image_size : the image size for the training and prediction
        distance_thres : the maximum distance accepted for recognition confidence
        auto_train_dist : the maximum distance accepted for auto recognition and training

        disp : the telegram distpacher
        classify_start_inline : the inline keyboar for the command /classify
        classify_start_msg : the message for the command /classify


    """

    def __init__(self, disp):
        """Init the class and start the telegram handlers"""

        Thread.__init__(self)

        self.stop_event = threading.Event()

        # ======RECOGNIZER VARIABLES======
        self.image_size = (200, 200)
        self.distance_thres = 95
        self.auto_train_dist = 80
        self.classifier = load_pkl(pt.model)
        self.face_thrs = 0.75
        self.faces_idx = 0
        self.clf_flag = 2  # 0 if svm, 1 if knn, 2 distance
        self.X, self.y = build_dataset()

        # ======TELEGRAM VARIABLES========
        self.disp = disp
        self.current_face = None

        # Creating conversation handler
        conversation = ConversationHandler(
            [CallbackQueryHandler(self.new_face, pattern="/unknown_new")],
            states={
                1: [MessageHandler(Filters.text, self.get_new_name)]

            },
            fallbacks=[CallbackQueryHandler(self.end_callback, pattern="/end")]
        )

        # Custom inlinekeyboard and start message
        self.classify_start_inline = InlineKeyboardMarkup([
            [InlineKeyboardButton("See Faces", callback_data="/classify_see"),
             InlineKeyboardButton("Save Faces", callback_data="/classify_save")],
            [InlineKeyboardButton("Filter Faces", callback_data="/classify_filter"),
             InlineKeyboardButton("Exit", callback_data="/end")]])
        self.classify_start_msg = "Welcome, here you can choose what you want to do"

        # adding everything to the bot
        disp.add_handler(conversation)
        disp.add_handler(CallbackQueryHandler(self.see_faces, pattern="/classify_see"))
        disp.add_handler(CallbackQueryHandler(self.send_faces, pattern="/view_face"))
        disp.add_handler(CallbackQueryHandler(self.send_unknown_face, pattern="/classify_save"))
        disp.add_handler(CallbackQueryHandler(self.filter_images_tg, pattern="/classify_filter"))
        disp.add_handler(CallbackQueryHandler(self.move_known_face, pattern="/unknown_known"))
        disp.add_handler(CallbackQueryHandler(self.delete_unkwon_face, pattern="/unknown_del"))
        disp.add_handler(CommandHandler("classify", self.classify_start))
        disp.add_handler(CallbackQueryHandler(self.end_callback, pattern="/end"))

    def run(self):
        """Run the thread, first train the model then just be alive"""

        # updater.start_polling()

        self.filter_all_images()

        while True:
            # if the thread has been stopped
            if self.stopped():
                # save the recognizer
                self.classifier.save(pt.recognizer)
                return

    def stop(self):
        self.stop_event.set()

    def stopped(self):
        return self.stop_event.is_set()

    # ===================TELEGRAM=========================

    """The following commands all have the same parameters:
    bot (obj) : the telegram bot
    update (obj) : the current update (message, inline press...)"""

    def classify_start(self, bot, update):
        """
        Initial function for the classify stage
        :param bot: the telegram bot
        :param update: the update recieved
        :return:
        """

        update.message.reply_text(self.classify_start_msg, reply_markup=self.classify_start_inline)

    def see_faces(self, bot, update):
        """Function to choose what face the user want to see
         :param bot: the telegram bot
        :param update: the update recieved
        :return:"""

        # generate the inline keyboard with the custom callback
        inline = self.generate_inline_keyboard("/view_face ")

        # if there is no inline keyboard it means there are no saved faces
        if not inline:
            self.back_to_start(bot, update, "Sorry... no saved faces were found")
            return

        to_send = "What face do you want to see?"

        # edit the previous message
        bot.edit_message_text(
            chat_id=update.callback_query.message.chat_id,
            text=to_send,
            message_id=update.callback_query.message.message_id,
            parse_mode="HTML",
            reply_markup=inline
        )

    def send_faces(self, bot, update):
        """Function to send all the photo of a specific subdir
         :param bot: the telegram bot
        :param update: the update recieved
        :return:"""

        # get the param
        s_name = update.callback_query.data.split()[1]
        user_id = update._effective_user.id

        # check if the name is in the faces dir
        dir_name = [elem for elem in os.listdir(pt.FACES_DIR) if s_name in elem]
        if len(dir_name) == 0:
            self.back_to_start(bot, update, "Sorry no face found with name " + s_name)
            return

        # take the dir name
        dir_name = dir_name[0] + "/"
        # for every image in the dir
        for image in glob.glob(pt.join(pt.FACES_DIR, pt.join(dir_name, '*.png'))):
            # open the image and send it
            with open(image, "rb") as file:
                bot.sendPhoto(user_id, file)

        self.end_callback(bot, update, calling=False)

        self.classify_start(bot, update.callback_query)

    def send_unknown_face(self, bot, update):
        """Function to send an unknown face (in the Unknown dir)
         :param bot: the telegram bot
        :param update: the update recieved
        :return:"""

        def resize_image(image_file, desired_size):
            """
            Resize an image to a desired dimension
            :param image_file: the image to be resized
            :param desired_size: the desired dimension
            :return: the new path to the image
            """

            new_path = pt.join(pt.RESOURCES_DIR, "to_send.png")

            img = Image.open(image_file)

            old_size = img.size  # old_size[0] is in (width, height) format

            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

            img.resize(new_size, Image.ANTIALIAS)
            img.save(new_path)

            return new_path

        to_choose = glob.glob(pt.join(pt.UNK_DIR, '*.png'))

        if len(to_choose) == 0:
            self.back_to_start(bot, update, "Sorry, no photos found")
            return

        image = random.choice(to_choose)
        self.current_face = image
        callback_image = image.split("/")[-1]
        user_id = update._effective_user.id

        inline = self.generate_inline_keyboard("/unknown_known " + callback_image + " ",
                                               InlineKeyboardButton("New",
                                                                    callback_data="/unknown_new " + callback_image),
                                               InlineKeyboardButton("Delete",
                                                                    callback_data="/unknown_del " + callback_image),
                                               InlineKeyboardButton("Exit", callback_data="/end " + callback_image))

        bot.delete_message(
            chat_id=update.callback_query.message.chat_id,
            message_id=update.callback_query.message.message_id,

        )
        to_send = "You can either choose one of the known faces, create a new one or delete the photo\nThere are currently " \
                  "" + str(len(to_choose)) + " photos to be classified"

        image = resize_image(image, 840)

        with open(image, "rb") as file:
            bot.sendPhoto(user_id, file,
                          caption=to_send,
                          reply_markup=inline)

        os.remove(image)

    def move_known_face(self, bot, update):
        """Function to move a known face from Unknown dir to face_dir
         :param bot: the telegram bot
        :param update: the update recieved
        :return:"""

        # the param has the format:  image_name dir_name
        param = update.callback_query.data.split()
        image_name = pt.join(pt.UNK_DIR, param[1])
        dir_name = param[2]

        # get the length of the images in the directory
        dir_name = pt.join(pt.FACES_DIR, get_name_dir(dir_name))
        idx = len([name for name in os.listdir(dir_name)])
        # generate new image name
        new_image_name = pt.join(dir_name, f"image_{idx}.png")

        # delete the photo message
        self.end_callback(bot, update, calling=False)

        # move the image
        try:
            os.rename(image_name, new_image_name)
        except FileNotFoundError:
            update.callback_query.message.reply_text("Photo not found")

        self.classify_start(bot, update.callback_query)

    def delete_unkwon_face(self, bot, update):
        """Function to delete a photo from the unknown dir
         :param bot: the telegram bot
        :param update: the update recieved
        :return:"""

        image = update.callback_query.data.split()[1]
        image = pt.join(pt.UNK_DIR, image)
        self.end_callback(bot, update, calling=False)

        try:
            os.remove(image)
        except FileNotFoundError:
            update.callback_query.message.reply_text("Photo not found!")
            return

        update.callback_query.message.reply_text("Photo deleted")

        self.classify_start(bot, update.callback_query)

    def new_face(self, bot, update):
        """Function to ask the user the name of the new subject
         :param bot: the telegram bot
        :param update: the update recieved
        :return:"""

        to_send = "Please insert the subject name\nYou have just one chance so be careful"

        bot.edit_message_caption(
            chat_id=update.callback_query.message.chat_id,
            caption=to_send,
            message_id=update.callback_query.message.message_id,
            parse_mode="HTML")

        return 1

    def get_new_name(self, bot, update):
        """Function to get a user name and create a folder
         :param bot: the telegram bot
        :param update: the update recieved
        :return:"""

        image_name = self.current_face
        face_name = update.message.text
        if not face_name or not image_name:
            update.message.reply_text("I told you to be carefull!")
            return ConversationHandler.END

        if glob.glob(pt.join(pt.FACES_DIR, "*_" + face_name)):
            update.message.reply_text("You cannot use the same name for two faces")
            return ConversationHandler.END

        self.move_image(image_name, face_name)
        update.message.reply_text("Done! You can now check the image under " + face_name)

        self.classify_start(bot, update)
        return ConversationHandler.END

    def back_to_start(self, bot, update, msg):
        """
        Send the initial start message
        :param bot: the telegram bot
        :param update: the update recieved
        :param msg: Custom message
        :return:
        """

        bot.edit_message_text(
            chat_id=update.callback_query.message.chat_id,
            text=msg,
            message_id=update.callback_query.message.message_id,
            parse_mode="HTML",
            reply_markup=self.classify_start_inline
        )

    def end_callback(self, bot, update, calling=True):
        """
        End the calssify deleting the message
        :param bot: the telegram bot
        :param update: the update recieved
        :param calling: if the button EXIT has not been pressed calling will be false
        :return:
        """

        bot.delete_message(
            chat_id=update.callback_query.message.chat_id,
            message_id=update.callback_query.message.message_id,

        )

        # look for new images in the Unknown direcotory and delete recognized ones
        if calling:
            msg = update.callback_query.message.reply_text("Updating recognizer...", parse_mode="HTML")

            self.train_model()

            bot.edit_message_text(
                chat_id=update.callback_query.message.chat_id,
                text="...Done",
                message_id=update.callback_query.message.message_id,
                parse_mode="HTML",
                reply_markup=self.classify_start_inline
            )

            time.sleep(1.3)

            bot.delete_message(
                chat_id=msg.chat_id,
                message_id=msg.message_id,

            )

    def filter_images_tg(self, bot, update):

        def edit_msg(to_send):
            # edit the previous message
            bot.edit_message_text(
                chat_id=update.callback_query.message.chat_id,
                text=to_send,
                message_id=update.callback_query.message.message_id,
                parse_mode="HTML",
                reply_markup=self.classify_start_inline
            )

        self.filter_all_images(stream=edit_msg)

    # ===================RECOGNIZER=========================

    def train_model(self):
        """
        Trains a k-nearest neighbors classifier for face recognition.

        :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
        :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
        :param verbose: verbosity of training
        :return: returns knn classifier that was trained on the given data.
        """

        def analize(X, Y):
            enc = preprocessing.LabelEncoder()
            y_enc = enc.fit_transform(y)

            pca = PCA()
            projected = pca.fit_transform(X, Y)

            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')

            plt.savefig(pt.join(pt.ANALISYS_DIR, "variance_components"))

            plt.clf()
            fig = plt.figure()
            ax = Axes3D(fig)

            ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                       c=y_enc, edgecolor='red', alpha=0.5, s=30,
                       cmap=plt.cm.get_cmap('viridis', 10))

            plt.savefig(pt.join(pt.ANALISYS_DIR, "3d_plot"))

            plt.clf()

            plt.scatter(projected[:, 0], projected[:, 1],
                        c=y_enc, edgecolor='none', alpha=0.5,
                        cmap=plt.cm.get_cmap('viridis', 10))
            plt.xlabel('component 1')
            plt.ylabel('component 2')
            plt.colorbar()
            plt.savefig(pt.join(pt.ANALISYS_DIR, "2d_plot"))

            plt.clf()

            fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                                     subplot_kw={'xticks': [], 'yticks': []},
                                     gridspec_kw=dict(hspace=0.1, wspace=0.1))
            for i, ax in enumerate(axes.flat):
                ax.imshow(pca.components_[i].reshape(8, 16), cmap='bone')

            plt.savefig(pt.join(pt.ANALISYS_DIR, "faces"))

        def build_classifier_knn(X, y):

            n_neighbors = int(round(math.sqrt(len(X)))) * 2
            print("Chose n_neighbors automatically:", n_neighbors)

            clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="brute", weights='uniform')
            clf.fit(X, y)
            return clf

        def build_classifier_svm(X, y):
            clf = svm.SVC()
            clf.fit(X, y)
            return clf

        self.X, self.y = build_dataset()

        if not len(self.X): return

        # Determine how many neighbors to use for weighting in the KNN classifier

        analize(self.X, self.y)

        if self.clf_flag:

            clf = build_classifier_svm(self.X, self.y)
        else:
            clf = build_classifier_knn(self.X, self.y)

        # save classifier to current one
        self.classifier = clf

        # Save the trained KNN classifier
        dump_pkl(clf, pt.model)

        print("Model trained")

    def predict(self, img):
        """
        Recognizes faces in given image using a trained KNN classifier
        :param img: an image to be recognized
        :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
               of mis-classifying an unknown person as a known one.
        :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
            For faces of unrecognized persons, the name 'unknown' will be returned.
        """

        # Load a trained KNN model if not already loaded
        if self.classifier is None:
            self.classifier = load_pkl(pt.model)

        # if no recognizer raise error
        if self.classifier is None:
            raise FileNotFoundError("Recognizer not loaded, cannot make prediction")

        # Load image file and find face locations
        face_locations = face_recognition.face_locations(img)

        # If no faces are found in the image, return an empty result.
        if len(face_locations) == 0:
            return None

        # Find encodings for faces in the test image
        faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_locations)

        def predict_knn():
            """
            Use knn to perform the prediction
            :return:
            """

            # Use the KNN model to find the best matches for the test face
            closest_distances = self.classifier.kneighbors(faces_encodings, n_neighbors=3)
            are_matches = [closest_distances[0][i][0] for i in range(len(face_locations))]

            # Predict classes and remove classifications that aren't within the threshold
            return [{'pred': pred, 'bbs': loc, 'conf': rec} for pred, loc, rec in
                    zip(self.classifier.predict(faces_encodings), face_locations, are_matches)]

        def predict_svm():
            """
            Use svm to perform the prediction
            :return:
            """
            predictions = self.classifier.predict(faces_encodings)
            are_matches = ["-" for elem in predictions]

            # Predict classes and remove classifications that aren't within the threshold
            return [{'pred': pred, 'bbs': loc, 'conf': rec} for pred, loc, rec in
                    zip(self.classifier.predict(faces_encodings), face_locations, are_matches)]

        def predict_distance():
            """
            Use custom algortihm to perform the prediction
            :return:
            """

            results = []

            for idx in range(len(faces_encodings)):
                distances = face_recognition.face_distance(self.X, faces_encodings[idx])
                pred, measure = distances_algorithm(distances, self.y)

                res_dict = {
                    'pred': pred,
                    'bbs': face_locations[idx],
                    'conf': measure

                }
                results.append(res_dict)

            return results


        if self.clf_flag==0:

            return predict_svm()

        elif self.clf_flag==1:

            return predict_knn()

        else:
            return predict_distance()

    def predict_multi(self, imgs, save=False):
        """ Predict faces in multiple images
        :param imgs: list of images
        :return: list of triples (face_name, confidence,image) for every Different face in the image list
        """

        print("Predict multi started...")

        # if there are no images return
        if len(imgs) == 0: return None
        predictions = []
        cropped = []

        for img in imgs:
            # predict single image
            pred = self.predict(img)
            predictions.append(pred)

            # skip not predicted frames
            if pred is None:
                continue

            for p in pred:
                # get bbs and crop image
                (top, right, bottom, left) = p['bbs']
                cropped_img = img[top:bottom, left:right]
                cropped.append(cropped_img)

        # if save append found images in direcotry
        if save:
            self.add_image_write(cropped)

        print("...Predict multi ended")
        return predictions

    def find_faces(self, image, save=False, model="cnn"):
        """
        Find faces in image and return the first
        :param image: the image as PIL
        :param save: (bool) if to save the image in the UNK dir
        :return: either the cropped image or None
        """

        face_images = []

        # look for the locations in the images
        face_locations = face_recognition.face_locations(image, model=model)

        # for every location crop image and append
        for location in face_locations:
            top, right, bottom, left = location
            face = image[top:bottom, left:right]

            face_images.append(face)
        # if there are any faces
        if len(face_images):
            # if to save
            if save:
                self.add_image_write(face_images)
            # return faces
            return face_images[0]

        return None

    # ===================UTILS=========================

    def show_prediction_labels_on_image(self, img, predictions):
        """
        Shows the face recognition results visually.
        :param img: path to image to be recognized
        :param predictions: results of the predict function
        :return:
        """

        if predictions is None: return

        for pred in predictions:
            # Draw a box around the face using the Pillow module

            (top, right, bottom, left) = pred['bbs']

            name = pred['pred']
            try:
                conf = round(100 * pred['conf'], 1)
            except TypeError:
                conf = pred['conf']

            pt1 = (left, top)
            pt2 = (right, bottom)

            cv2.rectangle(img, pt1, pt2, color=(0, 255, 0),
                          thickness=2)

            cv2.putText(img, str(name), (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0))

            cv2.putText(img, f"conf:{conf}", (int(left), int(bottom) - 10), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0))

    def generate_inline_keyboard(self, callback_data, *args):
        """Generate an inline keyboard containing the names of the known faces plus any inlinebutton passed with args"""

        # get all the saved subjects names
        names = get_dir_subjects()

        # uf there are none return
        if len(names) == 0 and len(args) == 0:
            return False

        # print(names)

        # add the names to the inline button
        rows = []
        cols = []

        for name in names:

            # only three cols allowed
            if len(cols) == 3:
                rows.append(cols)
                cols = []
            # add the buttom to the row
            cols.append(InlineKeyboardButton(name, callback_data=callback_data + name))

        # if there was less than three faces append them
        if len(cols) > 0:
            rows.append(cols)
        if not rows:
            rows.append(cols)

        # if there are other buttons from args, append them
        if args: rows.append(args)

        inline = InlineKeyboardMarkup(rows)

        return inline

    def add_image_write(self, image_list, subject_name=""):

        # currently used only for the unknown directory
        subject_name = "Unknown"
        path = pt.join(pt.FACES_DIR, subject_name)

        print("Adding face images to unknown folder...")

        # look for the directory and create it if not present
        if not glob.glob(path):
            return False
            # self.add_folder(subject_name)

        # get the directory name
        _dir = get_name_dir(subject_name)

        if not _dir:
            print("No direcotry")
            return False
        else:
            _dir = pt.join(pt.FACES_DIR, _dir + "/")
        # get the length of the images in the directory
        idx = len([name for name in os.listdir(_dir) if "png" in name]) + 1

        for image in image_list:
            image_name = pt.join(_dir, f"image_{idx}.png")
            cv2.resize(image, self.image_size)
            cv2.imwrite(image_name, image)
            idx += 1

        print("...Done")

        # self.auto_train()

        # remove similar images
        # threading.Thread(target=self.filter_all_images).start()

        return True

    def move_image(self, image, subject_name):

        # look for the direcotry and create it if not present
        print(subject_name)
        subject_name = subject_name.strip()
        if not any(subject_name in x for x in os.listdir(pt.FACES_DIR)):
            add_folder(subject_name)

        # get the directory name
        _dir = get_name_dir(subject_name)

        if not _dir:
            return False
        else:
            _dir = pt.join(pt.FACES_DIR, _dir)

        # get the length of the images in the directory
        idx = len([name for name in os.listdir(_dir) if os.path.isfile(name)])

        image_name = pt.join(_dir, f"image_{idx}.png")

        os.rename(image, image_name)

        return True

    def filter_all_images(self, stream=None):
        """
        Remove all the images in the database which are similar to each other
        :return:
        """

        def print_stream(to_send):
            print(to_send)

        if stream is None:
            stream = print_stream

        stream("Filtering images...")

        # get a list of paths for every image in Faces
        img_paths = []
        for path, subdirs, files in os.walk(pt.FACES_DIR):
            for name in files:
                if "png" in name:
                    img_paths.append(os.path.join(path, name))

        stream(f"Found {len(img_paths)} images... Filtering")

        # read them all using opencv
        images = [cv2.imread(elem) for elem in img_paths]
        # get the indices to be removed
        to_remove = filter_similar_images(images)

        if not len(to_remove):
            stream("No image to remove")
            return

        # get the paths corresponding to the indices
        to_remove = operator.itemgetter(*to_remove)(img_paths)
        if not isinstance(to_remove, tuple): to_remove = [to_remove]
        # remove them
        for elem in to_remove:
            os.remove(elem)

        for sub in subdirs:
            rename_images_index(os.path.join(pt.FACES_DIR, sub))

        stream(f"Removed {len(to_remove)} images")


# ===================STATIC=========================


def distances_algorithm(distance, y, algorithm="lowestSum"):
    """
    Return a tuple of prediction, confidence given an algorithm
    :param distance: a list of floats associated with the distance between faces
    :param y: a list of persons of the same lenght of tdistancess
    :param algorithm: str, the type of the algoritmh to use
    :return:
    """

    normalized_distance = distance / distance.sum()

    def top_n(n=10):
        """
        Get the top n best distance and use a voting system to get the most probable result
        :param n: int, the top
        :return:
        """
        pred = Counter(y[distance.argsort()[:n]]).most_common(1)[0][0]
        measure = normalized_distance[y == pred].sum()

        return pred, measure

    def lowest_sum():
        """
        Perform a sum for every person in the KB and get the one with the minimum weighted sum
        :return:
        """
        categ = Counter(y)

        distance_dict = {}
        for key, val in categ.items():
            distance_dict[key] = normalized_distance[y == key].sum() / val

        pred = min(distance_dict.items(), key=operator.itemgetter(1))[0]
        measure = distance_dict[pred]

        return pred, measure

    if algorithm=="topN":
        return top_n()

    elif algorithm=="lowestSum":
        return lowest_sum()

    else:
        raise ValueError(f"algoritm '{algorithm}' not recognized")




def rename_images_index(path_to_dir):
    img_paths = []
    renames = []
    tmp_paths = []
    idx = 0
    for path, subdirs, files in os.walk(path_to_dir):
        for name in files:
            if "png" in name:
                img_paths.append(os.path.join(path, name))
                renames.append(os.path.join(path, f"image_{idx}.png"))
                tmp_paths.append(os.path.join(path, f"image_copy_{idx}.png"))
                idx += 1

    for original, tmp in zip(img_paths, tmp_paths):
        os.renames(original, tmp)

    for tmp, renamed in zip(tmp_paths, renames):
        os.renames(tmp, renamed)


def build_dataset():
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(pt.FACES_DIR):
        if not os.path.isdir(os.path.join(pt.FACES_DIR, class_dir)) or "Unknown" in class_dir:
            continue

        # save directory
        subject_dir = os.path.join(pt.FACES_DIR, class_dir)

        # load encodings
        encodings = load_pkl(pt.join(subject_dir, pt.encodings))
        if encodings is None: encodings = []

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(subject_dir):
            image = face_recognition.load_image_file(img_path)
            os.remove(img_path)

            # take the bounding boxes an the image size
            face_bounding_boxes = 0, 0, image.shape[0], image.shape[1]
            face_bounding_boxes = [face_bounding_boxes]

            # Add face encoding for current image to the training set
            encodings.append(
                face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes,
                                                num_jitters=10)[0])
            # y.append(class_dir.split("_")[-1])

        # save encodings
        dump_pkl(encodings, pt.join(subject_dir, pt.encodings))
        print(f"Encodings for {subject_dir} are {len(encodings)}")
        # update model
        X += encodings
        y += len(encodings) * [class_dir.split("_")[-1]]

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def filter_prediction_subjects(predictions):
    """
    Filter prediction and returns a list of found faces based on maximum confidence
    :param predictions: zipped list (predictions, images)
    :return: list of tuples (predicted name, croppped image)
    """

    # remove empty list
    filtered = [elem for elem in predictions if elem[0] is not None]

    # unzip list
    filtered, images = zip(*filtered)

    best_dict = {}
    # for every list of prediction in an image
    for idx in range(len(filtered)):

        # for every prediction in a list
        for jdx in range(len(filtered[idx])):

            # get the predicted face and the confidence
            pred = filtered[idx][jdx]['pred']
            conf = filtered[idx][jdx]['conf']

            try:
                # update value if confidence is more
                if best_dict[pred][0] < conf:
                    best_dict[pred] = (conf, idx, jdx)
            except KeyError:
                # append it otherwise
                best_dict[pred] = (conf, idx, jdx)

    to_return = []
    # for every best results
    for key, val in best_dict.items():
        # crop image and append it to list
        idx = val[1]
        jdx = val[2]
        top, right, bottom, left = filtered[idx][jdx]['bbs']
        cropped = images[idx][top:bottom, left:right]

        to_return.append((key, cropped))

    return to_return


def prepare_training_data():
    """Get the saved images from the Faces direcotry, treat them and return two lists with the same lenght:
    faces : list of images with faces in them
    labels : list of labels for each face """

    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = glob.glob(pt.join(pt.FACES_DIR, "s_*"))
    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:
        # print(dir_name)
        # get the subject label (number)
        label = int(dir_name.split("/")[-1].split("_")[1])

        # for every image in the direcotry append image,label
        for image_path in glob.glob(pt.join(dir_name, "*.png")):
            # read the image
            image = cv2.imread(image_path)
            # convert to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # append image
            faces.append(gray)
            labels.append(label)
            # remove image
            os.remove(image_path)

    return faces, labels


def add_folder(name):
    """Create a folder for the new person"""

    if not name in os.listdir(pt.FACES_DIR):
        # get how many folder there are in the faces dir
        idx = len(glob.glob(pt.join(pt.FACES_DIR, 's_*')))
        # generate the name
        name = "s_" + str(idx) + "_" + name
        # create the directory
        os.makedirs(pt.join(pt.FACES_DIR, name))


def get_name_dir(subject_name):
    for dir in os.listdir(pt.FACES_DIR):
        if subject_name in dir:
            return dir

    return False


def get_dir_subjects():
    """Function to get all the names saved in the faces direcotry"""

    s_names = []

    for name in glob.glob(pt.join(pt.FACES_DIR, 's_*')):
        s_names.append(name.split("_")[2])

    return s_names


def filter_similar_images(images, similar_thresh=0.94):
    """
    Filter from lis of images the ones which have a high similarity
    :param images: a list of np arrays
    :return: list of indices of the images to be removed
    """

    def rmse(img_1, img_2):
        """
        Run similarity measure between two iamges
        :param img_1:
        :param img_2:
        :return:
        """

        # get total measure
        dim_a = np.sum(img_1.shape)
        dim_b = np.sum(img_2.shape)

        # convert to PIL image
        img_1 = Image.fromarray(np.uint8(img_1))
        img_2 = Image.fromarray(np.uint8(img_2))

        # resize to same shape
        if dim_a < dim_b:
            img_2 = img_2.resize(img_1.size, Image.ANTIALIAS)
        else:
            img_1 = img_1.resize(img_2.size, Image.ANTIALIAS)

        # reconvert to numpy array
        img_1 = np.asarray(img_1)
        img_2 = np.asarray(img_2)

        # performa similarity measure

        a, b, _ = img_1.shape
        score = np.sqrt(np.sum((img_2 - img_1) ** 2) / float(a * b))
        max_val = max(np.max(img_1), np.max(img_2))
        min_val = min(np.min(img_1), np.min(img_2))
        return 1 - (score / (max_val - min_val))

    # remove images with zero dimension
    images = [img for img in images if not 0 in img.shape]
    to_pop = []

    for idx in trange(len(images) - 1):
        for jdx in range(idx + 1, len(images)):
            # measure similarity
            similarity = rmse(images[idx], images[jdx])
            # if it is more than thresh
            if similarity >= similar_thresh:
                to_pop.append(jdx)

    to_pop = list(set(to_pop))

    return to_pop

# # uncomment and add token to debug face recognition
# updater = Updater("545431258:AAHEocYDtLOQdZDCww6tQFSfq3p-xmWeyE8")
# disp = updater.dispatcher
# # #
# face = FaceRecognizer(disp)
# # face.start()
#
