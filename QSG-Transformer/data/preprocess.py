# -*- coding:utf-8 -*-

import os, json
import dgl
import dgl.function as fn
import stanza
from stanza.server import CoreNLPClient
import networkx as nx
import torch

def _graph(coref):
    token_list = []
    token2node = []
    node_list = {}
    edge_list = []
    query_node_list = []
    coref_comp_map = {}

    node_idx = 0

    # Make token_list for BART tokenization & Initialize token2node
    for sent in coref.sentence:
        sent_token = [t.word for t in sent.token]
        sent_token2node = [-1] * len(sent_token)
        
        token_list.append(sent_token)
        token2node.append(sent_token2node)

    # Coreference Resolution
    for coref_chain in coref.corefChain:
        node_list['coref' + str(node_idx)] = node_idx
        for mention in coref_chain.mention:
            sent_idx = mention.sentenceIndex
            begin_idx = mention.beginIndex
            end_idx = mention.endIndex
            
            coref_comp_map['coref' + str(node_idx)] = coref.sentence[sent_idx].token[begin_idx:end_idx][0].lemma
            
            for j in range(begin_idx, end_idx):
                token2node[sent_idx][j] = node_idx
            
            # Append node_idx to query_node_list
            if sent_idx == 0:
                query_node_list.append(node_idx)

        node_idx += 1
    
    # Combine compound token 
    for sent_idx, sent in enumerate(coref.sentence):
        for edge in sent.basicDependencies.edge:
            if edge.dep == 'compound':
                source_token_idx = edge.source - 1
                target_token_idx = edge.target - 1
                if token2node[sent_idx][source_token_idx] != -1 or token2node[sent_idx][target_token_idx] != -1:
                    continue
                node_list['compound' + str(node_idx)] = node_idx
                token2node[sent_idx][source_token_idx] = node_idx
                token2node[sent_idx][target_token_idx] = node_idx
                
                target_lemma = coref.sentence[sent_idx].token[target_token_idx].lemma
                source_lemma = coref.sentence[sent_idx].token[source_token_idx].lemma
                coref_comp_map['compound' + str(node_idx)] = target_lemma + " " + source_lemma

                if sent_idx == 0:
                    query_node_list.append(node_idx)

                node_idx += 1
    
    # Make other node list
    for sent_idx, sent in enumerate(coref.sentence):
        for token_idx, token in enumerate(sent.token):
            if token2node[sent_idx][token_idx] != -1:
                continue
            if not token.pos.isalpha():
                continue
            if token.pos == 'DT':
                continue
            
            noun_pos = ['NN', 'NNS', 'NNP', 'NNPS']
            
            if token.pos in noun_pos:
                lemma = token.lemma
                if lemma in node_list:
                    token2node[sent_idx][token_idx] = node_list[lemma]
                else:
                    node_list[token.lemma] = node_idx
                    token2node[sent_idx][token_idx] = node_idx

                    if sent_idx == 0:
                        query_node_list.append(node_idx)

                    node_idx += 1
                continue
            
            node_list[token.word + str(node_idx)] = node_idx
            coref_comp_map[token.word + str(node_idx)] = token.word

            token2node[sent_idx][token_idx] = node_idx

            if sent_idx == 0:
                query_node_list.append(node_idx)

            node_idx += 1            
    
    # Connect node via edge
    for sent_idx, sent in enumerate(coref.sentence):
        for edge in sent.basicDependencies.edge:
            if edge.dep == 'punct':
                continue
            source_token_idx = edge.source - 1
            target_token_idx = edge.target - 1
            if token2node[sent_idx][source_token_idx] == -1 or token2node[sent_idx][target_token_idx] == -1:
                continue
            
            source_node_idx = token2node[sent_idx][source_token_idx]
            target_node_idx = token2node[sent_idx][target_token_idx]
            
            if source_node_idx == target_node_idx:
                continue
            
            edge = (source_node_idx, target_node_idx)
            edge_list.append(edge)
    
    # Connect query root node & other sentences root node
    root_node_idx_list = []
    for sent_idx, sent in enumerate(coref.sentence):
        root_token_idx = sent.basicDependencies.root[0] - 1
        root_node_idx = token2node[sent_idx][root_token_idx]
        
        if root_node_idx == -1 and sent_idx == 0:
            root_node_idx_list.append(0)
        elif root_node_idx == -1 and sent_idx != 0:
            continue
        else:
            root_node_idx_list.append(root_node_idx)
    for i in range(1, len(root_node_idx_list)):
        edge = (root_node_idx_list[0], root_node_idx_list[i])
        edge_list.append(edge)
        
    # Others
    _token_list = []
    for temp in token_list:
        _token_list += temp
    
    _token2node = []
    for temp in token2node:
        _token2node += temp

    query_node_list = list(set(query_node_list))
    
    # Add self-loop
    node_len = max(_token2node) + 1
    for i in range(node_len):
        edge_list.append((i, i))

    node_list_keys = list(node_list.keys())
    node_list_values = list(node_list.values())
    node_list = {}

    for k, v in zip(node_list_keys, node_list_values):
        if k in coref_comp_map:
            node_list[v] = coref_comp_map[k]
        else:
            node_list[v] = k

    return _token_list, _token2node, edge_list, query_node_list, node_list

def _pagerank(edge, query_node_list, n_pr = 3, alpha = 0.15):
    src = [e[0] for e in edge]
    dst = [e[1] for e in edge]

    g = dgl.graph((src, dst))
    g = dgl.remove_self_loop(g)
    g = dgl.to_simple(g)
    g = dgl.to_bidirected(g)

    g.ndata['score'] = torch.ones(g.num_nodes(), dtype = torch.float) / g.num_nodes()
    degrees = g.out_degrees(g.nodes()).type(torch.float)
    
    teleport = torch.zeros(g.num_nodes())
    teleport[query_node_list] = 1. / len(query_node_list)
    
    for _ in range(n_pr):
        g.ndata['score'] = g.ndata['score'] / degrees
        g.update_all(message_func=fn.copy_src(src='score', out='m'),
                     reduce_func=fn.sum(msg='m', out='score'))
        g.ndata['score'] = alpha * teleport + (1 - alpha) * g.ndata['score']

    pr_score = g.ndata['score']
    pr_score = torch.where(pr_score == 0, torch.tensor(1e-10, dtype = torch.float), pr_score)

    pr_score = pr_score.tolist()
        
    return pr_score, src, dst

def _aux_label(src, dst, query_node_list):
    g = dgl.graph((src, dst))
    g = dgl.to_simple(g)
    g = dgl.to_bidirected(g)

    nx_g = dgl.to_networkx(g)

    length = dict(nx.all_pairs_shortest_path_length(nx_g))

    label = [10000] * len(length)

    for q in query_node_list:
        temp = length[q]

        for i, leng in temp.items():
            if leng < label[i]:
                label[i] = leng

    label = [l if l < 5 else 5 for l in label]

    return label

def setting():
    stanza.install_corenlp()

def process(filename = 'train.json'):
    path = os.path.join('./Debatepedia', filename)

    with open(path, 'r') as f:
        data = json.load(f)

    with CoreNLPClient(
        annotators = ['coref'],
        timeout = 30000000,
        memory = '6G'
    ) as client:

        outputs = []

        for dp in data:
            text = dp['query'] + " " + dp['document']
            coref = client.annotate(text)

            word_list, word2node, edge_list, _query_node_list, node_list = _graph(coref)
            pr_score, _src, _dst = _pagerank(edge_list, _query_node_list)
            aux_label = _aux_label(_src, _dst, _query_node_list)

            dp['word'] = word_list
            dp['word2node'] = word2node
            dp['edge'] = edge_list
            dp['pr_score'] = pr_score
            dp['aux_label'] = aux_label

            outputs.append(dp)

    with open(filename, 'w') as f:
        f.write(json.dumps(outputs))

if __name__ == '__main__':
    setting()
    process('train.json')
    process('valid.json')
    process('test.json')