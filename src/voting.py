
from collections import Counter
from typing import Dict, List, Optional, Tuple

class TwoStageVoter:
    def __init__(self, temperatures: List[float], tau_intra: float = 0.8, tau_cross: float = 1.0):
        self.temps = temperatures
        self.tau_intra = tau_intra
        self.tau_cross = tau_cross
        self.per_temp_answers: Dict[float, List[str]] = {t: [] for t in self.temps}

    def add_answer(self, temp: float, answer: str):
        self.per_temp_answers[temp].append(answer)

    def _intra_confident(self, answers: List[str]) -> Tuple[bool, Optional[str], float]:
        if not answers:
            return False, None, 0.0
        cnt = Counter(answers)
        top_ans, top_cnt = cnt.most_common(1)[0]
        conf = top_cnt / max(1, len(answers))
        return (conf >= self.tau_intra), top_ans, conf

    def step(self) -> Tuple[bool, Optional[str], Dict[float, Dict]]:
        votes = []
        debug = {}
        for t in self.temps:
            confident, top_ans, conf = self._intra_confident(self.per_temp_answers[t])
            debug[t] = {"confident": confident, "top": top_ans, "conf": conf, "n": len(self.per_temp_answers[t])}
            if not confident:
                return False, None, debug
            votes.append(top_ans)

        cross = Counter(votes)
        top_ans, top_cnt = cross.most_common(1)[0]
        if (top_cnt / len(self.temps)) >= self.tau_cross:
            return True, top_ans, debug
        return False, None, debug
