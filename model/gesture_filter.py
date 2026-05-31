"""
时域投票过滤器 (Temporal Voting Filter)
论文表4中的 GestureCommandFilter 实现

滑窗投票机制：8帧窗口内至少5帧一致且置信度>0.75才触发指令，附带1秒冷却时间。
"""
from collections import deque, Counter
import time


class GestureCommandFilter:
    """
    基于滑窗投票的指令过滤器

    工作机制：
    1. 维护长度为 window 的预测历史队列
    2. 统计队列中置信度 ≥ threshold 的标签
    3. 当同一标签出现 ≥ min_votes 次，且冷却时间已过，触发指令
    4. 无手势(no_gesture)不被触发
    """

    def __init__(self, window: int = 8, min_votes: int = 5,
                 threshold: float = 0.75, cooldown: float = 1.0):
        """
        Args:
            window: 滑窗帧数 (论文: 8)
            min_votes: 最少一致票数 (论文: 5)
            threshold: 置信度阈值 (论文: 0.75)
            cooldown: 冷却时间/秒 (论文: 1.0)
        """
        self.history = deque(maxlen=window)
        self.min_votes = min_votes
        self.threshold = threshold
        self.cooldown = cooldown
        self.last_trigger_time = 0.0
        self.last_command = None

        # 统计信息
        self.total_triggers = 0
        self.rejected_count = 0

    def update(self, label: str, prob: float, now: float = None) -> str | None:
        """
        输入一帧的预测结果，返回是否触发指令

        Args:
            label: 预测的标签名 (e.g. "palm", "fist")
            prob: 预测置信度 (0-1)
            now: 当前时间戳（秒），默认使用 time.time()

        Returns:
            触发的指令名（如 "light_on"），未触发则返回 None
        """
        if now is None:
            now = time.time()

        # 加入历史队列
        self.history.append((label, prob))

        # 筛选置信度达标的预测
        labels_above_threshold = [
            x[0] for x in self.history
            if x[1] >= self.threshold
        ]

        # 数量不足或冷却时间内不触发
        if len(labels_above_threshold) < self.min_votes:
            self.rejected_count += 1
            return None

        if now - self.last_trigger_time < self.cooldown:
            self.rejected_count += 1
            return None

        # 多数投票
        label_counts = Counter(labels_above_threshold)
        most_common_label, count = label_counts.most_common(1)[0]

        # 票数不够或为"无手势"则不触发
        if count < self.min_votes or most_common_label == "no_gesture":
            self.rejected_count += 1
            return None

        # 触发指令
        self.last_trigger_time = now
        self.last_command = most_common_label
        self.total_triggers += 1
        self.history.clear()  # 触发后清空历史，避免重复触发

        from config import COMMAND_MAP
        return COMMAND_MAP.get(most_common_label, "idle")

    def get_vote_status(self) -> dict:
        """
        获取当前投票状态（用于可视化）

        Returns:
            dict with keys: labels_count, is_ready, cooldown_remaining, progress_pct
        """
        labels_above = [
            x[0] for x in self.history
            if x[1] >= self.threshold
        ]
        label_counts = Counter(labels_above)
        most_common = label_counts.most_common(1)
        current_label, current_count = most_common[0] if most_common else ("—", 0)

        cooldown_remaining = max(0, self.cooldown - (time.time() - self.last_trigger_time))
        is_ready = (
            current_count >= self.min_votes
            and cooldown_remaining == 0
            and current_label != "no_gesture"
        )

        return {
            "current_label": current_label,
            "current_count": current_count,
            "min_votes": self.min_votes,
            "window_size": self.history.maxlen,
            "is_ready": is_ready,
            "cooldown_remaining": cooldown_remaining,
            "progress_pct": min(100, int(current_count / self.min_votes * 100)),
            "history_size": len(self.history),
        }

    def reset(self):
        """重置过滤器状态"""
        self.history.clear()
        self.last_trigger_time = 0.0
        self.last_command = None
