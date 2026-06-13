import time
import random
import logging
import threading
import sys

logger = logging.getLogger("legalassist.chaos")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('[CHAOS MONKEY] %(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

class ChaosMonkey:
    """
    A chaos testing tool designed to simulate failure modes in the Legalassist-AI backend.
    Use this strictly in staging/development environments to test microservice resilience.
    """
    
    def __init__(self, probability: float = 0.2):
        """
        :param probability: The likelihood (0.0 to 1.0) of a chaotic event occurring per cycle.
        """
        self.probability = probability
        self.running = False
        
    def _simulate_latency(self):
        delay = random.uniform(2.0, 10.0)
        logger.warning(f"Injecting artificial latency... Network requests will stall for {delay:.2f}s.")
        time.sleep(delay)
        
    def _simulate_service_crash(self):
        service = random.choice(["DocumentParser", "VectorDatabase", "AuthService", "CeleryWorker"])
        logger.error(f"FATAL: Simulating catastrophic crash in {service}!")
        # In a real environment, this might run `docker stop <container>` or kill a process
        
    def _simulate_db_timeout(self):
        logger.warning("Simulating database connection pool exhaustion/timeout.")
        
    def _chaos_loop(self):
        logger.info(f"Chaos Monkey unleashed with {self.probability*100}% attack probability.")
        while self.running:
            time.sleep(5) # Evaluate every 5 seconds
            
            if random.random() < self.probability:
                attack = random.choice([
                    self._simulate_latency,
                    self._simulate_service_crash,
                    self._simulate_db_timeout
                ])
                attack()
                
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._chaos_loop, daemon=True)
            self.thread.start()
            
    def stop(self):
        self.running = False
        logger.info("Chaos Monkey contained.")

if __name__ == "__main__":
    monkey = ChaosMonkey(probability=0.4)
    monkey.start()
    
    try:
        # Keep main thread alive while chaos ensues
        for _ in range(6):
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        monkey.stop()
