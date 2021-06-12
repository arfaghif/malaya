from herpetologist import check_type
from malaya.supervised import transformer as load_transformer
from malaya.function.parse_dependency import DependencyGraph
from malaya.model.tf import KnowledgeGraph

_transformer_availability = {
    'base': {
        'Size (MB)': 246,
        'Quantized Size (MB)': 63.8,
        'BLEU': 0.8572,
        'Suggested length': 256,
    },
    'large': {
        'Size (MB)': 632,
        'Quantized Size (MB)': 161,
        'BLEU': 0.8595,
        'Suggested length': 256,
    },
}


def parse_from_dependency(tagging, indexing,
                          subjects=[['flat', 'subj', 'nsubj', 'csubj']],
                          relations=[['acl', 'xcomp', 'ccomp', 'obj', 'conj', 'advcl'], ['obj']],
                          objects=[['obj', 'compound', 'flat', 'nmod', 'obl']],
                          get_networkx=True):
    """
    Generate knowledge graphs from dependency parsing.

    Parameters
    ----------
    tagging: List[Tuple(str, str)]
        `tagging` result from dependency model.
    indexing: List[Tuple(str, str)]
        `indexing` result from dependency model.
    subjects: List[List[str]], optional
        List of dependency labels for subjects.
    relations: List[List[str]], optional
        List of dependency labels for relations.
    objects: List[List[str]], optional
        List of dependency labels for objects.
    get_networkx: bool, optional (default=True)
            If True, will generate networkx.MultiDiGraph.

    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - Transformer BASE parameters.
        * ``'large'`` - Transformer LARGE parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: Dict[result, G]
    """

    if get_networkx:
        try:
            import pandas as pd
            import networkx as nx
        except BaseException:
            logging.warning(
                'pandas and networkx not installed. Please install it by `pip install pandas networkx` and try again. Will skip to generate networkx.MultiDiGraph'
            )
            get_networkx = False

    def combined(r):
        results, last = [], []
        for i in r:
            if type(i) == tuple:
                last.append(i)
            else:
                for no, k in enumerate(last):
                    if k[1] == i[0][1]:
                        results.append(last[:no] + i)
                        break
        results.append(last)
        return results

    def get_unique(lists):
        s = set()
        result = []
        for l in lists:
            str_s = str(l)
            if str_s not in s:
                result.append(l)
                s.add(str_s)
        return result

    def get_longest(lists):
        r = []
        for l in lists:
            if len(l) > len(r):
                r = l
        return r

    def postprocess(r, labels=['subject', 'relation', 'object']):
        if all([l not in r for l in labels]):
            return

        for l in labels:
            if len(r[l]) == 0:
                return

            r[l] = ' '.join([i[0] for i in r[l]])

        return r

    result = []
    for i in range(len(tagging)):
        result.append(
            '%d\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_'
            % (i + 1, tagging[i][0], int(indexing[i][1]), tagging[i][1])
        )

    d_object = DependencyGraph('\n'.join(result), top_relation_label='root')
    results = []
    for i in range(1, len(indexing), 1):
        if d_object.nodes[i]['rel'] == 'root':
            subjects_, relations_ = [], []
            for s in subjects:
                s_ = d_object.traverse_children(i, s, initial_label=[d_object.nodes[i]['rel']])
                s_ = combined(s_)
                s_ = [c[1:] for c in s_]
                subjects_.extend(s_)
            for s in relations:
                s_ = d_object.traverse_children(i, s, initial_label=[d_object.nodes[i]['rel']])
                s_ = combined(s_)
                relations_.extend(s_)
            subjects_ = get_unique(subjects_)
            subject = get_longest(subjects_)
            relations_ = get_unique(relations_)

            for relation in relations_:
                objects_ = []
                k = relation[-1][1]
                for s in objects:
                    s_ = d_object.traverse_children(k, s, initial_label=[d_object.nodes[k]['rel']])
                    s_ = combined(s_)
                    objects_.extend(s_)
                objects_ = get_unique(objects_)
                obj = get_longest(objects_)
                if obj[0][0] == relation[-1][0] and len(obj) == 1:
                    results.append({'subject': subject, 'relation': relation[:-1], 'object': relation[-1:]})
                else:
                    if obj[0][0] == relation[-1][0]:
                        obj = obj[1:]
                    results.append({'subject': subject, 'relation': relation, 'object': obj})

    post_results = []
    for r in results:
        r = postprocess(r)
        if r:
            post_results.append(r)

    r = {'result': post_results}

    if get_networkx:
        df = pd.DataFrame(post_results)
        G = nx.from_pandas_edgelist(
            df,
            source='subject',
            target='object',
            edge_attr='relation',
            create_using=nx.MultiDiGraph(),
        )
        r['G'] = G

    return r


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 200k test set.'
    )


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load transformer to generate knowledge graphs in triplet format from texts,
    MS text -> EN triplet format.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - Transformer BASE parameters.
        * ``'large'`` - Transformer LARGE parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.KnowledgeGraph class
    """
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.knowledge_graph.available_transformer()`.'
        )

    return load_transformer.load(
        module='knowledge-graph-generator',
        model=model,
        encoder='sentencepiece',
        model_class=KnowledgeGraph,
        quantized=quantized,
        **kwargs
    )
