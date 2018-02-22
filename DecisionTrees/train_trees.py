import numpy as np
# from visualize_trees import *
from helper_functions import *
from copy import deepcopy
import pickle
from scipy.io import loadmat


# HELPER FUNCTIONS (need to be local)
def node_id():
    """
    generator function - is used by mark_nodes(tree_) to assign nodes to tree_
    """
    for node in range(1000):
        yield node


def mark_nodes(tree_):
    """
    uses node_id() generator to assign nodes to tree (depth first)
    """
    tree_.node = next(node_i)
    if len(tree_.kids) is not 0:
        for kid in tree_.kids:
            mark_nodes(tree_.kids[kid])


def pop_list(tree_):
    """
    populates a list with tree_ nodes
    the format of the resulting tree_list is
    [ #node (int), attribute that is split (int), kids (list), label (list), rate of positives (float) ]
    """
    kids = []
    if len(tree_.kids) is not 0:
        for i in tree_.kids:
            kids.append(tree_.kids[i].node)
    tree_list.append([tree_.node, tree_.op, kids, tree_.label, tree_.rate])
    if len(tree_.kids) is not 0:
        for i in tree_.kids:
            pop_list(tree_.kids[i])



if __name__ == "__main__":

    # LOAD DATA
    data = loadmat('Data/cleandata_students.mat')
    # data = loadmat('Data/noisydata_students.mat')
    combine_trees = 'performance'  # 'random'

    x = data['x']
    y = data['y']
    N = x.shape[0]
    attributes = np.array(range(0, int(x.shape[1])))


    # RUN k-FOLD CROSS-VALIDATION
    K = 10

    # errors for multiclass
    errors_pruned = []
    errors_unpruned = []

    # confusion matrices for multiclass
    conf_mat_multi_pruned = []
    conf_mat_multi_unpruned = []

    # all trained trees
    all_unpruned_trees = []
    all_pruned_trees = []

    # confusion matrices for individual trees
    conf_mat_bin_pruned = []
    conf_mat_bin_unpruned = []

    for fold_id in range(1, K+1):

        conf_mat_bin_pruned.append([])
        conf_mat_bin_unpruned.append([])

        print('Fold %d of %d' % (fold_id, K))

        # SPLIT DATA PER FOLD
        xtrain, ytrain, xtest, ytest = make_k_fold_set(x, y, K, i=fold_id)

        unpruned_trees = {}
        pruned_trees = {}

        for emotion in range(1, 7):

            print('emotion %d' % emotion)

            # TRAIN TREE
            binary_targets = target_binary_vector(ytrain, emotion)
            new_tree = decision_tree_learning(xtrain, attributes, binary_targets)

            # PREDICT BINARIES BASED ON TRAINED TREE
            ypred = new_tree.predict(xtest)

            # CALCULATE PERFORMANCE STATISTICS FOR UNPRUNED TREE
            test_binary_targets = target_binary_vector(ytest, emotion)
            recall, precision = recall_precision(test_binary_targets, ypred)
            f1_score = f_score(recall, precision)
            error_ = error(test_binary_targets, new_tree.predict(xtest))

            unpruned_trees[emotion] = {'tree': new_tree, 'recall': recall,
                                      'precision': precision, 'f1_score': f1_score, 'error': error_}
            all_unpruned_trees.append(unpruned_trees)

            conf_mat_bin_unpruned[fold_id-1].append(
                confusion_mat(test_binary_targets, ypred))

            # COPY TRAINED TREE OBJECT
            trees = deepcopy(unpruned_trees)

            tree_ = trees[emotion]['tree']
            error_ = trees[emotion]['error']

            # MAKE TREE TO LIST FORMAT
            node_i = node_id()

            mark_nodes(tree_)

            tree_list = []

            pop_list(tree_)

            # START PRUNING
            # FOR A MAXIMUM OF ROUNDS EQUAL TO THE NUMBER OF NODES
            for i in range(len(tree_list)):

                error_pruned = {}
                # FOR EACH NODE IN THE TREE
                for node_ in range(1, len(tree_list)):

                    tree_list2 = deepcopy(tree_list)

                    del tree_list2[node_]

                    ypred_list = test_tree_list(xtest, tree_list2)

                    error_pruned[node_] = error(test_binary_targets, ypred_list)

                # FIND BEST NODE TO PRUNE
                node_to_prune = min(error_pruned, key=error_pruned.get)
                error_pruned_ = error_pruned[node_to_prune]

                # CHECK THAT PRUNING IMPROVES ERROR RATE
                if error_pruned_ < error_:

                    del tree_list[node_to_prune]

                    error_ = error_pruned_

                # IF IT DOES NOT; STOP PRUNING (greedy)
                else:
                    break

            # GET PERFORMANCE STATISTICS FOR FINAL PRUNED TREE
            ypred_list = test_tree_list(xtest, tree_list)

            recall, precision = recall_precision(test_binary_targets, ypred_list)#test_tree_list(xtest, tree_list))
            f1_score = f_score(recall, precision)

            # ADD CONFUSION MATRICES FOR INDIVIDUAL TREES
            conf_mat_bin_pruned[fold_id - 1].append(
                confusion_mat(test_binary_targets, ypred_list))

            # ADD FINAL PRUNED TREE TO DICT OF PRUNED TREES
            pruned_trees[emotion] = {'tree_list': tree_list, 'error': error_,
                                     'precision': precision, 'recall': recall, 'f1_score': f1_score}

        # FINISHED PRUNING TREES
        # NOW WE HAVE A DICT OF 6 PRUNED TREES (one per emotion)
        # SAVE
        all_pruned_trees.append(pruned_trees)

        # CLASSIFY TEST SET WITH PRUNED AND UNPRUNED TREES AND GET SUMMARY STATISTICS
        y_unpruned = testTrees2(unpruned_trees, xtest, combine_trees)
        errors_unpruned.append(error(ytest, y_unpruned))
        conf_mat_multi_unpruned.append(confusion_mat(ytest, y_unpruned))

        y_pruned = testTrees2(pruned_trees, xtest, combine_trees)
        errors_pruned.append(error(ytest, y_pruned))
        conf_mat_multi_pruned.append(confusion_mat(ytest, y_pruned))


    def get_results_per_emotion(trees, statistic):
        return [np.mean([tree[em][statistic] for tree in trees]) for em in range(1, 7)]

    def mirror(m):
        return np.array([[m[1,1], m[1, 0]],
                         [m[0,1], m[0, 0]]])

    # UNPRUNED
    print('UNPRUNED')
    print('Results below are averaged over the 10 folds, for each emotion')
    precisions = get_results_per_emotion(all_unpruned_trees, 'precision')
    recalls = get_results_per_emotion(all_unpruned_trees, 'recall')
    f1_scores = get_results_per_emotion(all_unpruned_trees, 'f1_score')
    class_rate = [1 - i for i in get_results_per_emotion(all_unpruned_trees, 'error')]
    print('Average Precision :', precisions)
    print('Average Recall :', recalls)
    print('Average F1_score :', f1_scores)
    print('Average Classification Rate :', class_rate)
    av_conf_mat_bin_unpruned = [average_confusion_matrix([conf_mat_bin_unpruned[i][j] for i in range(10)])
                                                         for j in range(6)]
    av_conf_mat_bin_unpruned = [mirror(conf_) for conf_ in av_conf_mat_bin_unpruned]
    print('Average confusion matrices :')
    print(av_conf_mat_bin_unpruned)

    print('Results below are averaged over the 10 folds, for the multiclass case')
    average_confusion_matrix(conf_mat_multi_unpruned)
    print('Average Classification error :', np.mean(errors_unpruned))
    print('Classification errors over the 10 folds :', errors_unpruned)
    print('Confusion Matrix :')
    print(average_confusion_matrix(conf_mat_multi_unpruned))

    # PRUNED
    print('PRUNED')
    print('Results below are averaged over the 10 folds, for each emotion')
    precisions = get_results_per_emotion(all_pruned_trees, 'precision')
    recalls = get_results_per_emotion(all_pruned_trees, 'recall')
    f1_scores = get_results_per_emotion(all_pruned_trees, 'f1_score')
    class_rate = [1 - i for i in get_results_per_emotion(all_pruned_trees, 'error')]
    print('Average Precision :', precisions)
    print('Average Recall :', recalls)
    print('Average F1_score :', f1_scores)
    print('Average Classification Rate :', class_rate)
    av_conf_mat_bin_pruned = [average_confusion_matrix([conf_mat_bin_pruned[i][j] for i in range(10)])
                                                         for j in range(6)]
    av_conf_mat_bin_pruned = [mirror(conf_) for conf_ in av_conf_mat_bin_pruned]
    print('Average confusion matrices :')
    print(av_conf_mat_bin_pruned)

    print('Results below are averaged over the 10 folds, for the multiclass case')
    print('Average Classification error :', np.mean(errors_pruned))
    print('Classification errors over the 10 folds :', errors_pruned)
    print('Confusion Matrix :')
    print(average_confusion_matrix(conf_mat_multi_unpruned))

    # PICKLE
    # print(pruned_trees)
    # print(len(pruned_trees))
    # pickle.dump(pruned_trees, open("trained_trees_clean_data.p", "wb"))
