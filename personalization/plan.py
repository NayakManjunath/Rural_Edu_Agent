# personalization/plan.py
"""
Simple rule-based learning plan generator: given topics, time, bandwidth -> weekly plan.
"""
from typing import Dict, Any, List
from utils.config import BANDWIDTH_PROFILES

def estimate_content_size(resource_type: str, duration_min: int) -> int:
    """
    crude bytes estimate: video ~ 500KB/min at low res, audio ~ 50KB/min, text small
    """
    if resource_type == "video":
        return 500 * 1024 * duration_min
    if resource_type == "audio":
        return 50 * 1024 * duration_min
    return 10 * 1024  # text

def make_weekly_plan(student_profile: Dict[str,Any], topics: List[str], bandwidth_tier: str="low") -> Dict[str,Any]:
    """
    Returns plan with daily items sized to bandwidth tier.
    """
    daily_time = student_profile.get("daily_minutes", 30)
    kbps = BANDWIDTH_PROFILES.get(bandwidth_tier, BANDWIDTH_PROFILES["low"])
    plan = {"student": student_profile.get("name", "unknown"), "bandwidth_tier": bandwidth_tier, "days": []}
    # simple round-robin topics to 7 days
    for day in range(7):
        topic = topics[day % max(1, len(topics))]
        # choose resource type
        if kbps < 100:
            resource_type = "text"
            estimated_size = estimate_content_size("text", 0)
        elif kbps < 1000:
            resource_type = "audio"
            estimated_size = estimate_content_size("audio", int(daily_time/2))
        else:
            resource_type = "video"
            estimated_size = estimate_content_size("video", int(daily_time/2))
        plan["days"].append({
            "day": day+1,
            "topic": topic,
            "resource_type": resource_type,
            "minutes": int(daily_time/len(topics)) if topics else daily_time,
            "estimated_size_bytes": estimated_size
        })
    return plan

# Example:
# make_weekly_plan({"name":"Raju", "daily_minutes":30}, ["math","science"], "medium")
