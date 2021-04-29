#!/usr/bin/env python

import argparse
import cv2
import os
import sys
from matplotlib import pyplot as plt

def get_groundtruth(gt_file):
    """
    Read a ground truth file and outputs a dictionary
    mapping queries to the set of relevant results (plus a list of all images).
    """
    gt = {}
    allnames = set()
    with open(gt_file, "r") as file:
        for line in file:
            imname = line.strip()
            allnames.add(imname)
            imno = int(imname[:-len(".jpg")])
            if imno % 100 == 0:
                gt_results = set()
                gt[imname] = gt_results
            else:
                gt_results.add(imname)

    return (allnames, gt)

def parse_results_file(fname):
    """
    Go through the results file and 
    return them in suitable structures
    """
    res = {}
    with open(fname, "r") as file:
        for line in file:
            fields = line.split()
            query_name = fields[0]
            ranks   = [int(rank) for rank in fields[1::2]]
            imnames = [im for im in fields[2::2]]
            res[query_name] = list(zip(ranks, imnames))

    return res

def compute_AP(query, gt):
    """
    Compute the average precision of one search.
    query = Ordered list of image filenames
    gt = Set with the relevant images for this query
    """
    ap = 0.0
    nrel = len(gt)
    curr_k = 1
    curr_rel = 0

    for imname in query:
        
        # Checking if the returning result is relevant to the query
        if imname in gt:
            curr_rel += 1
            ap += float(curr_rel) / float(curr_k)
        curr_k += 1

    return ap / nrel

def compute_mAP_from_file(results_file, gt_file):
    """
    Compute mAP from a file using the INRIA Holidays dataset format
    results_file = Results file following the indicated format
    gt_file = Ground truth file. Typically, 'holidays_images.dat'.
    """
    # Reading GT file
    (gt_names, gt) = get_groundtruth(gt_file)

    # Parsing results file
    results = parse_results_file(results_file)

    # Sum of AP's
    sum_ap = 0.0
    nqueries = 0

    # Processing each query
    for query_name,query_results in results.items():

        # Checking if the current query is in the dataset
        if query_name not in gt:
            print('Unknown query: %s' % query_name)
            return -1

        # Get GT for this query
        gt_results=gt.pop(query_name)

        # Sorting in ascending order
        query_results.sort()

        # Filtering the results        
        query_results_filt = []
        for _,res_name in query_results:
            #  Checking if the returned name is correct
            if res_name not in gt_names:
                print("Image name '%s' not in Holidays" % res_name)
                return -1
            
            # Checking if any of the results is the query itself
            if res_name == query_name:
                continue
            
            query_results_filt.append(res_name)      
        
        ap = compute_AP(query_results_filt, gt_results)
        sum_ap += ap
        nqueries += 1
    
    if gt:
        # Some queries left
        print("No result for queries: ", gt.keys())
        return -1
    
    return sum_ap / nqueries

def compute_mAP(results, gt_file):
    """
    Compute mAP from a resulting dictionary
    results = Dictionary containing, for each query, the ordered list of retrieved images.
              Example: {'100100.jpg': ['100101.jpg', '100102.jpg']}
    gt_file = Ground truth file. Typically, 'holidays_images.dat'.
    """
    # Reading GT file
    (gt_names, gt) = get_groundtruth(gt_file)

    # Sum of AP's
    sum_ap = 0.0
    nqueries = 0

    # Processing each query
    # for query_name,results in parse_results(results_file):
    for query_name,query_results in results.items():        

        # Checking if the current query is in the dataset
        if query_name not in gt:
            print('Unknown query: %s' % query_name)
            return -1

        # Get GT for this query
        gt_results=gt.pop(query_name)

        # Filtering the results        
        query_results_filt = []
        for res_name in query_results:
            #  Checking if the returned name is correct
            if res_name not in gt_names:
                print("Image name '%s' not in Holidays" % res_name)
                return -1
            
            # Checking if any of the results is the query itself
            if res_name == query_name:
                continue
            
            query_results_filt.append(res_name)      
        
        ap = compute_AP(query_results_filt, gt_results)
        sum_ap += ap
        nqueries += 1
    
    if gt:
        # Some queries left
        print("No result for queries: ", gt.keys())
        return -1
    
    return sum_ap / nqueries

class ResultViz(object):
    '''
    Class to visualize graphically the results of a retrieval procedure
    '''
    
    def __init__(self, q_names, q_imgs, t_names, t_imgs):
        '''
        Constructor. It receives lists with query and train names and images.
        '''
        self.q_names = q_names
        self.q_imgs = q_imgs
        self.t_names = t_names
        self.t_imgs = t_imgs
        

    def show_results(self, results, nqueries = 3, ntrains = 2):
        '''
        Show the results of an image retrieval process.

        - results: A dictionary containing, for each query image, an ordered list of the retrieved images. See compute_mAP for an example.
        - nqueries: Number of query images to show from the results.
        - ntrains: Number of retrieved images to show for each query.
        '''
        fig, axs = plt.subplots(nrows=nqueries, ncols=ntrains + 1, figsize=(20, 20),
                                squeeze=False, subplot_kw={'xticks': [], 'yticks': []})
  
        # Hiding all axes by default
        for ax in axs.flatten():
            ax.set_visible(False)
  
        id_query = 0
        for query_name, query_results in results.items():

            if id_query == nqueries:
                break
        
            # Showing query image
            qimg = self.q_imgs[self.q_names.index(query_name)]
            axs[id_query, 0].imshow(cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB), aspect=0.5)
            axs[id_query, 0].set_title(query_name)
            axs[id_query, 0].axis('scaled')
            axs[id_query, 0].set_visible(True)
                        
            # Showing ntrains best results
            for img_id, t_name in enumerate(query_results[:ntrains]):
                timg = self.t_imgs[self.t_names.index(t_name)]
                axs[id_query, img_id + 1].imshow(cv2.cvtColor(timg, cv2.COLOR_BGR2RGB), aspect=0.5)
                axs[id_query, img_id + 1].set_title(t_name)
                axs[id_query, img_id + 1].axis('scaled')
                axs[id_query, img_id + 1].set_visible(True)
    
            id_query += 1

        fig.tight_layout()
        fig.show()

    def show_one_result(self, results, query_name, ntrains = 2):
        '''
        Show the results of an image retrieval process for just one query.
  
        - results: A dictionary containing, for each query image, an ordered list of the retrieved images. See compute_mAP for an example.
        - query_name: Name of the query image to be shown.
        - ntrains: Number of retrieved images to show.
        '''
        
        fig, axs = plt.subplots(nrows=1, ncols=ntrains + 1, figsize=(20, 20),
                          squeeze=False, subplot_kw={'xticks': [], 'yticks': []})
                          
        # Hiding all axes by default
        for ax in axs.flatten():
            ax.set_visible(False)
  
        # Showing query image
        qimg = self.q_imgs[self.q_names.index(query_name)]
        axs[0, 0].imshow(cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB), aspect=0.5)    
        axs[0, 0].set_title(query_name)
        axs[0, 0].axis('scaled')
        axs[0, 0].set_visible(True)

        # Showing ntrains best results
        for img_id, t_name in enumerate(results[query_name][:ntrains]):
            timg = self.t_imgs[self.t_names.index(t_name)]
            axs[0, img_id + 1].imshow(cv2.cvtColor(timg, cv2.COLOR_BGR2RGB), aspect=0.5)
            axs[0, img_id + 1].set_title(t_name)
            axs[0, img_id + 1].axis('scaled')    
            axs[0, img_id + 1].set_visible(True)  
        
        fig.tight_layout()
        fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="Results file")
    args = parser.parse_args()

    results_file = args.results_file

    # Testing computation of mAP from a file
    m_ap = compute_mAP_from_file(results_file, 'holidays_images.dat')
    print("mAP for %s: %.5f" % (results_file, m_ap))

    # Testing computation of mAP from a dictionary
    
    # Parsing results file
    results = parse_results_file(results_file)
    results_new = {}
    for k,v in results.items():
        l = []
        for _,img in v:
            l.append(img)
        results_new[k] = l
    
    m_ap = compute_mAP(results_new, 'holidays_images.dat')
    print("mAP for %s: %.5f" % (results_file, m_ap))
