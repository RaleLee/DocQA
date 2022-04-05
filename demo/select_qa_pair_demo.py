from typing import List

import torch.nn as nn
from sentence_transformers import SentenceTransformer, util


class SearchCachedQAListDemo(nn.Module):
    def __init__(
            self, top: int, recall_rate: float, top_rate: float,
            recall_model_path: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            nli_model_path: str = 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
    ):
        """
        Init a search cached QA list demo with sentenceBERT.
        In case of chinese doc, using a multilingual version
        :param top: return rank top n
        :param recall_rate: a select rate for recall confidence
        :param top_rate: a select rate for top1 confidence
        :param recall_model_path: choose model version for sentenceBERT
        :param nli_model_path: choose model version for NLI model.
        """
        super(SearchCachedQAListDemo, self).__init__()
        self.model = SentenceTransformer(recall_model_path)
        self.top = top
        self.recall_rate = recall_rate
        self.top_rate = top_rate
        self.nli_model = SentenceTransformer(nli_model_path)

    def forward(self, user_queries: List[str], cached_queries: List[str], return_score: bool = False) -> dict:
        """
        Compute the cosine similarity between the user queries and system cached queries
        :param user_queries: list of user queries
        :param cached_queries: list of system cached queries
        :param return_score: if True, return the similarity dot_score at the same time
        :return: A dict contains the top cached queries and dot_score
        """
        user_queries_embedding = self.model.encode(user_queries, convert_to_tensor=True, normalize_embeddings=True)
        cached_queries_embedding = self.model.encode(cached_queries, convert_to_tensor=True, normalize_embeddings=True)
        # Recall
        # compute dot dot_score
        # return tensor convert to list
        dot_scores = util.dot_score(user_queries_embedding, cached_queries_embedding).cpu().numpy().tolist()

        rank_ret = []
        for dot_score in dot_scores:
            tmp_score = dot_score[:]
            rank = []
            for _ in range(self.top):
                ms = max(tmp_score)
                if ms < self.recall_rate:
                    break
                idx = tmp_score.index(ms)
                tmp_score[idx] = 0
                rank.append(cached_queries[idx])
            rank_ret.append(rank)

        ret = {'sentence_rank': rank_ret}
        if return_score:
            ret['rank_scores'] = dot_scores

        # Select TOP 1
        nli_user = self.nli_model.encode(user_queries, convert_to_tensor=True, normalize_embeddings=True)
        nli_cache = [self.nli_model.encode(r, convert_to_tensor=True, normalize_embeddings=True)
                     if len(r) > 0 else None for r in rank_ret]
        top = []
        for i, us in enumerate(nli_user):
            if nli_cache[i] is None:
                top.append(None)
                continue
            nli_score = util.dot_score(us, nli_cache[i]).squeeze(0).cpu().numpy().tolist()
            top_score = max(nli_score)
            if top_score < self.top_rate:
                top.append(None)
                continue
            top.append((rank_ret[i][nli_score.index(top_score)], top_score))
        ret['top'] = top
        return ret


if __name__ == '__main__':
    search_demo = SearchCachedQAListDemo(top=3, recall_rate=0.6, top_rate=0.7)
    # Change user questions and system cached questions here
    user = ["故宫在哪个城市", "怎么去天安门", "北京有什么好玩的", "今年清明节放假么", "疫情什么时候开始的"]
    sys_cached = ["故宫门票怎么预约", "故宫门票多少钱", "天安门广场怎么走", "故宫养心殿勤政亲贤匾额是由谁写的", "故宫博物院在哪"]

    result = search_demo(user, sys_cached, return_score=True)
    # see the ranking result
    print("Recall list")
    sen_ranks = result['sentence_rank']
    for u, sen in zip(user, sen_ranks):
        print("User's query: " + u)
        print("Recalls: " + str(sen))

    # see the top
    print("Find Top1")
    top1 = result['top']
    for u, t in zip(user, top1):
        print("User's query: " + u + " Find: " + (t[0] + " score: " + str(t[1]) if t is not None else "None"))

    # see the scores
    print("Recall Scores")
    scores = result['rank_scores']
    for u, score in zip(user, scores):
        print("User's query: " + u)
        for sys_c, sc in zip(sys_cached, score):
            print(sys_c + " " + str(sc))
        print()
