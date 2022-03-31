from typing import List

import torch.nn as nn
from sentence_transformers import SentenceTransformer, util


class SearchCachedQAListDemo(nn.Module):
    def __init__(
            self, top: int,
            model_path: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    ):
        """
        Init a search cached QA list demo with sentenceBERT.
        In case of chinese doc, using a multilingual version
        :param top: return rank top n
        :param model_path: choose model version for sentenceBERT.
        """
        super(SearchCachedQAListDemo, self).__init__()
        self.model = SentenceTransformer(model_path)
        self.top = top

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

        # compute dot dot_score
        # return tensor convert to list
        dot_scores = util.dot_score(user_queries_embedding, cached_queries_embedding).cpu().numpy().tolist()

        rank_ret = []
        for dot_score in dot_scores:
            tmp_score = dot_score[:]
            rank = []
            for _ in range(self.top):
                idx = tmp_score.index(max(tmp_score))
                tmp_score[idx] = 0
                rank.append(cached_queries[idx])
            rank_ret.append(rank)

        ret = {'sentence_rank': rank_ret}
        if return_score:
            ret['rank_scores'] = dot_scores
        return ret


if __name__ == '__main__':
    search_demo = SearchCachedQAListDemo(top=3)
    user = ["故宫在哪个城市", "怎么去天安门"]
    sys_cached = ["故宫门票怎么预约", "故宫门票多少钱", "天安门广场怎么走", "故宫养心殿勤政亲贤匾额是由谁写的", "故宫博物院在哪"]

    result = search_demo(user, sys_cached, return_score=True)
    # see the ranking result
    sen_ranks = result['sentence_rank']
    for sen in sen_ranks:
        print(sen)

    # see the scores
    scores = result['rank_scores']
    for u, score in zip(user, scores):
        print("User's query: " + u)
        for sys_c, sc in zip(sys_cached, score):
            print(sys_c + " " + str(sc))
        print()
