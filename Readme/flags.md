# Flags

This class handles the various flags used with the telegram command
*/flags*. It implements a father-child linked list made of dictionaries.
Each flag has a one or multiple parents andchildrens.

Since each instance if a flag, so its value is either true or false, the
setting of a certain instance to false backtrack to all its children
setting it to false. 

This is done in oreder to perform automatic dis/activation of certain
flags, for example the *darknet* functionality cannot work if the
*video* flag is False (that is because there are no frames saved in
memory to perform segmentation onto).
