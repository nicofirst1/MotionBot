import os

import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from face_recognition.face_detection_cli import image_files_in_folder
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing, neighbors, svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Classes.Face_recognizer import distances_algorithm
from Path import Path as pt
from Utils.serialization import load_pkl, dump_pkl


class DataAnalysis():

    def __init__(self, x, y):

        self.x = x
        self.y = y

    def set_data(self, x, y):

        self.x = x
        self.y = y

    def knn_analysis(self, projected, encoded, n_neighbors=None):

        def accuracy_knn(x, y):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return round(accuracy_score(y_test, y_pred) * 100, 2)

        if n_neighbors is None:
            n_neighbors = int(round(np.math.sqrt(len(projected))))

        projected = projected[:, :2]
        print("Chose n_neighbors automatically:", n_neighbors)
        X_train, X_test, y_train, y_test = train_test_split(projected, encoded, test_size=0.33, random_state=42)
        h = .02  # step size in the mesh
        for weights in ['uniform', 'distance']:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X_train, y_train)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X_train[:, 0].min() - 0.05, X_train[:, 0].max() + 0.05
            y_min, y_max = X_train[:, 1].min() - 0.05, X_train[:, 1].max() + 0.05
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap="viridis")

            accuracy = accuracy_knn(self.x, encoded)

            # Plot also the training points
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="viridis",
                        edgecolor='k', s=20)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("%d-Class classification (k = %i, w = '%s', acc = %d)"
                      % (len(set(encoded)), n_neighbors, weights, accuracy))

            plt.savefig(pt.join(pt.ANALISYS_DIR, f"knn_{weights}"))
            plt.clf()

    def pca_analysis(self, encoded_label):
        pca = PCA()
        projected = pca.fit_transform(self.x, self.y)

        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')

        plt.savefig(pt.join(pt.ANALISYS_DIR, "variance_components"))

        plt.clf()
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                   c=encoded_label, edgecolor='red', alpha=0.5, s=30,
                   cmap=plt.cm.get_cmap('viridis', 10))

        plt.savefig(pt.join(pt.ANALISYS_DIR, "3d_plot"))

        plt.clf()

        plt.scatter(projected[:, 0], projected[:, 1],
                    c=encoded_label, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('viridis', 10))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar()
        plt.savefig(pt.join(pt.ANALISYS_DIR, "2d_plot"))

        plt.clf()

        return projected

    def svm_analysis(self, X,Y):
        fignum = 1
        def accuracy_svm(x, y):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            clf = svm.SVC(kernel='linear', C=1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return round(accuracy_score(y_test, y_pred) * 100, 2)

        acc=accuracy_svm(X,Y)

        X = X[:, :2]

        indx=np.argwhere(Y<=2).squeeze()
        Y=Y[indx]
        X=X[indx]



        # fit the model
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(X, Y)

        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (clf.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors (margin away from hyperplane in direction
        # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
        # 2-d.
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin

        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(fignum, figsize=(4, 3))
        plt.clf()
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                    edgecolors='k')

        plt.axis('tight')
        x_min = x.min()-0.5
        x_max = x.max()+0.5
        y_min = -0.5
        y_max = 0.5

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.xticks(())
        plt.yticks(())
        fignum = fignum + 1

        plt.savefig(pt.join(pt.ANALISYS_DIR, f"svm"))
        plt.clf()

    def analyze(self):
        enc = preprocessing.LabelEncoder()
        encoded_label = enc.fit_transform(self.y)
        print("Executin PCA")
        projected = self.pca_analysis(encoded_label)
        print("Executing KNN")
        self.knn_analysis(projected, encoded_label)
        print("Executin SVM")
        self.svm_analysis(projected,encoded_label)

    def compare_scores(self):
        enc = preprocessing.LabelEncoder()
        encoded_label = enc.fit_transform(self.y)


        def accuracy_svm(x, y):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            clf = svm.SVC(kernel='linear', C=1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return round(accuracy_score(y_test, y_pred) * 100, 2)

        def accuracy_knn(x, y):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            clf = neighbors.KNeighborsClassifier(22, weights="uniform")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return round(accuracy_score(y_test, y_pred) * 100, 2)

        def accuracy_top_n(x,y):

            y_pred=[]
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            for idx in range(len(X_test)):
                distances = face_recognition.face_distance(X_train, X_test[idx])
                pred, measure = distances_algorithm(distances, y_train, algorithm="topN")

                y_pred.append(pred)

            return round(accuracy_score(y_test, y_pred) * 100, 2)

        def accuracy_lowes_sum(x, y):

            y_pred = []
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            for idx in range(len(X_test)):
                distances = face_recognition.face_distance(X_train, X_test[idx])
                pred, measure = distances_algorithm(distances, y_train, algorithm="lowestSum")

                y_pred.append(pred)

            return round(accuracy_score(y_test, y_pred) * 100, 2)

        for idx in range(2,len(set(encoded_label))+1):
            indx = np.argwhere(encoded_label <= idx).squeeze()

            svm_acc=accuracy_svm(x[indx],encoded_label[indx])
            knn_acc=accuracy_knn(x[indx],encoded_label[indx])
            topn_acc=accuracy_top_n(x[indx],encoded_label[indx])
            lowest_acc=accuracy_lowes_sum(x[indx],encoded_label[indx])


            print(f"Accuracies for {idx} classes:\nSvm = {svm_acc}\nKnn = {knn_acc}\nTopN = {topn_acc}\nLowesSum ="\
            f"{lowest_acc}\n\n")




def build_dataset_analysis():
    """
    build the dataset
    :return: X,Y as numpy arrays
    """
    x = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(pt.FACES_DIR):
        if not os.path.isdir(os.path.join(pt.FACES_DIR, class_dir)) or "Unknown" in class_dir:
            continue

        # save directory
        subject_dir = os.path.join(pt.FACES_DIR, class_dir)

        # load encodings
        encodings = load_pkl(pt.join(subject_dir, pt.encodings))
        if encodings is None:
            encodings = []

        # Loop through each training image for the current person
        errors = 0
        for img_path in image_files_in_folder(subject_dir):
            image = face_recognition.load_image_file(img_path)
            os.remove(img_path)

            # take the bounding boxes an the image size
            face_bounding_boxes = 0, image.shape[1], image.shape[0], 0
            face_bounding_boxes = [face_bounding_boxes]

            try:
                encode = \
                    face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes, num_jitters=10)[0]

                encodings.append(encode)

            except IndexError:
                print(f"out of range error number {errors}")
                errors += 1
                pass

        # save encodings
        dump_pkl(encodings, pt.join(subject_dir, pt.encodings))
        print(f"Encodings for {subject_dir} are {len(encodings)}")
        # update model
        x += encodings
        y += len(encodings) * [class_dir.split("_")[-1]]

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y


if __name__ == '__main__':
    x, y = build_dataset_analysis()
    analysis = DataAnalysis(x, y)
    analysis.compare_scores()
