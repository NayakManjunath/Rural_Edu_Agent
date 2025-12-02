# utils/api_limiter.py

import time

class APILimiter:
    def __init__(self, max_calls_per_min=10, max_calls_per_session=50):
        self.max_per_min = max_calls_per_min
        self.max_per_session = max_calls_per_session
        self.call_times = []
        self.session_calls = 0

    def allowed(self):
        now = time.time()

        # remove timestamps older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]

        # check limits
        if len(self.call_times) >= self.max_per_min:
            return False, "rate_limit"
        if self.session_calls >= self.max_per_session:
            return False, "session_limit"

        return True, None

    def record_call(self):
        now = time.time()
        self.call_times.append(now)
        self.session_calls += 1
