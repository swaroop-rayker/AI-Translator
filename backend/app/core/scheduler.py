import os
import redis
import logging

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# If running on Host (not inside Docker), adjust hostname
if not os.path.exists("/.dockerenv") and "redis:6379" in REDIS_URL:
    REDIS_URL = REDIS_URL.replace("redis:6379", "localhost:6379")

class GPUScheduler:
    def __init__(self):
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {REDIS_URL}. Fallback to in-memory scheduler. Error: {e}")
            self.redis_client = {}
            self.redis_available = False

    def acquire_gpu_lock(self, job_id: str, lease_seconds: int = 600) -> bool:
        """
        Attempts to acquire the GPU lock for a specific job_id.
        Returns True if successful, False if already locked.
        """
        if self.redis_available:
            try:
                # Set key if it doesn't exist (NX) with expiry (PX/EX)
                acquired = self.redis_client.set("gpu_lock", job_id, ex=lease_seconds, nx=True)
                if acquired:
                    self.redis_client.set("gpu_active_job_id", job_id)
                    logger.info(f"GPU lock acquired successfully for Job: {job_id}")
                    return True
                else:
                    current_owner = self.redis_client.get("gpu_lock")
                    logger.warning(f"GPU lock acquisition failed for Job: {job_id}. Current owner: {current_owner}")
                    return False
            except Exception as e:
                logger.error(f"Redis error in acquire_gpu_lock: {e}")
                # Fallback to in-memory dictionary-based lock
                self.redis_client = {"gpu_lock": job_id, "gpu_active_job_id": job_id}
                return True
        else:
            if not self.redis_client.get("gpu_lock"):
                self.redis_client["gpu_lock"] = job_id
                self.redis_client["gpu_active_job_id"] = job_id
                return True
            return self.redis_client["gpu_lock"] == job_id

    def release_gpu_lock(self, job_id: str) -> bool:
        """
        Releases the GPU lock if the current owner matches the job_id.
        """
        if self.redis_available:
            try:
                current_owner = self.redis_client.get("gpu_lock")
                if current_owner == job_id:
                    self.redis_client.delete("gpu_lock")
                    self.redis_client.delete("gpu_active_job_id")
                    logger.info(f"GPU lock released for Job: {job_id}")
                    return True
                logger.warning(f"GPU lock release skipped. Requestor {job_id} is not owner (Current: {current_owner})")
                return False
            except Exception as e:
                logger.error(f"Redis error in release_gpu_lock: {e}")
                self.redis_client = {}
                return True
        else:
            if self.redis_client.get("gpu_lock") == job_id:
                self.redis_client.pop("gpu_lock", None)
                self.redis_client.pop("gpu_active_job_id", None)
                return True
            return False

    def get_gpu_status(self) -> dict:
        """
        Returns the current lock owner and lock status.
        """
        if self.redis_available:
            try:
                owner = self.redis_client.get("gpu_lock")
                return {
                    "is_locked": owner is not None,
                    "active_job_id": owner,
                    "redis_connected": True
                }
            except Exception as e:
                logger.error(f"Redis error in get_gpu_status: {e}")
                return {"is_locked": False, "active_job_id": None, "redis_connected": False}
        else:
            owner = self.redis_client.get("gpu_lock")
            return {
                "is_locked": owner is not None,
                "active_job_id": owner,
                "redis_connected": False
            }

    def renew_gpu_lock(self, job_id: str, lease_seconds: int = 300) -> bool:
        """
        Renews the lease of the lock for the current owner.
        """
        if self.redis_available:
            try:
                current_owner = self.redis_client.get("gpu_lock")
                if current_owner == job_id:
                    self.redis_client.expire("gpu_lock", lease_seconds)
                    return True
                return False
            except Exception as e:
                logger.error(f"Redis error in renew_gpu_lock: {e}")
                return False
        return True

# Singleton instance
gpu_scheduler = GPUScheduler()
