#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@authors: Xian Fan Mallory
Contacting email: fan@cs.fsu.edu
"""

import sys
import argparse
import numpy as np

# tree is an array of treenode
class treenode():
    def __init__(self, id_):
        # root's ID is 0, root has two children
        self.id = id_
        self.tuple=[]
        # the edge is the one above the node; edge ID is the same as node id. 
        self.edge_length = 0
        # root's parent ID is -1
        self.parentID = -1
        # root's depth is 1
        self.depth_ = -1
        self.perc = -1
        self.if_leaf = True
    def setTuple(self, a, b):
        self.tuple = [a, b]
    def getTuple(self):
        return self.tuple
    def getID(self):
        return self.id
    def getDepth(self):
        return self.depth_
    def getPerc(self):
        return self.perc
        
# store the tree by edges
def save_tree(T, out_f):
    f = open(out_f, "w")
    for i in range(len(T)):
        leaf = 0
        if T[i].if_leaf:
            leaf = 1
        # edge ID which is the same as the child node ID, parent ID, child node ID
        f.write("\t".join([str(T[i].id), str(T[i].parentID), str(T[i].id), str(T[i].edge_length), str(T[i].tuple[1] - T[i].tuple[0]), str(leaf)]) + "\n")
    f.close()

def is_in(a, range_):
    if a >= range_[0] and a < range_[1]:
        return True
    return False

def gen_tree(Beta, Alpha, treeWidth):
    tree_W = treeWidth
   
    #Alpha = float(Alpha)
    #Beta = float(Beta)
    #Delta = float(Delta)
    # add a root (node 0) to the tree
    # edge length (there are at most 2*n - 1))
    #            | CN0
    #          node 0
    #        / CN1   \ CN2
    #    node 1    node 2
    ti = np.random.exponential(1,2*tree_W-1)
    Ui = np.random.uniform(0.0,1.0,tree_W-1)
    Bi = np.random.beta(float(Alpha+1),float(Beta+1),tree_W-1)

    #Normalizing the branch lengths
    ti[0] = np.random.exponential(1)
    summation = 0
    for t in ti:
        summation += t
    
    for i in range(0,len(ti)):
        ti[i]=float(ti[i])/float(summation)

    #Contructing the phylogeny
    Tree = []
    Tree.append(treenode(0))
    Tree[0].perc = 1
    Tree[0].parentID = -1
    Tree[0].edge_length = ti[0]
    Tree[0].if_leaf = False
    Tree[0].depth_ = 1
    Tree[0].tuple=[0,1]

    Tree.append(treenode(1))
    Tree.append(treenode(2))
       
    # set parent ID
    Tree[1].parentID = 0
    Tree[2].parentID = 0
    # set percentage
    Tree[1].perc = Bi[0]
    Tree[2].perc = 1 - Bi[0]
    Tree[1].if_leaf = True
    Tree[2].if_leaf = True
    # set depth
    Tree[1].depth_ = 2
    Tree[2].depth_ = 2


    Tree[1].tuple=[0,Bi[0]]
    Tree[2].tuple=[Bi[0],1]
    Tree[1].edge_length = ti[1]
    Tree[2].edge_length = ti[2]

    leaf_num = 2
    node_number = 2

    j = 1
    while leaf_num < tree_W:
        for i in range(len(Tree)):
            node = Tree[i]
            # for SNV CNA project, we have to use the normal Beta Splitting model. The modified way does not record the percentage of the CNA cells, and the percentage inside the tuple does not reflect the true CNA cells. 06132022. Change back to normal.
            if node.if_leaf and is_in(Ui[j], node.getTuple()) : # and (not tree.is_dead) :
            # change is_in to not is_in so that the clone sizes may have a big contrast from each other when beta is small
            #if node.if_leaf and not is_in(Ui[j], node.getTuple()) : # and (not tree.is_dead) :
                # expand on this clone
                Tree[i].if_leaf = False
                leaf_num += 1
                this_id = node.getID()

                # add two more leaves
                node_number+=2
                Tree.append(treenode(node_number-1))
                Tree.append(treenode(node_number))
                # set parent id
                Tree[node_number].parentID = this_id
                Tree[node_number-1].parentID = this_id

                # set depth
                depth = node.depth_ + 1

                Tree[node_number - 1].depth_ = depth
                Tree[node_number].depth_ = depth

                # determine the percentage from the Beta splitting and parents' percentage
                perc = node.getPerc()

                Tree[node_number - 1].perc = perc * float(Bi[j])
                Tree[node_number].perc = perc * (1 - float(Bi[j])) 
                Tree[node_number - 1].if_leaf = True
                Tree[node_number].if_leaf = True

                #The new intervals are assigned here
                a,b = node.getTuple()
                middle = float(Bi[j])*float((float(b)-float(a)))+float(a)
                Tree[node_number-1].tuple=[a,middle]
                Tree[node_number].tuple=[middle,b]
                Tree[node_number-1].edge_length = ti[node_number-1]
                Tree[node_number].edge_length = ti[node_number]

                print("Done with adding nodes " + str(node_number - 1) + " and " + str(node_number))
                j = j + 1

                break
    return Tree
    
if len(sys.argv) <= 1:
    print("""
    This generates a tree according to Beta splitting model. 
    Usage: python main.py -r [dir] -n [num_leaves] -B [beta]
        -F (--treewidth)    Number of leaves of the tree. [16]
        -B (--Beta)         Beta value in Beta splitting model. 0.5 is evenly distributed. [0.3]
        -o (--out-file)     Output file. [tree.csv]
    """)
    sys.exit(0)

parser = argparse.ArgumentParser(description='This outputs a tree from Beta splitting model.')
parser.add_argument('-F', '--treewidth', default=16)
parser.add_argument('-B', '--Beta', default=0.3)
parser.add_argument('-o', '--out-file', default="cna_tree.tsv")

args = parser.parse_args()
treeWidth = int(args.treewidth)
Beta = float(args.Beta)
out_f = args.out_file

T = gen_tree(Beta, 0.5, treeWidth)
save_tree(T, out_f)
