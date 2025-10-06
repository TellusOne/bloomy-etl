from datetime import datetime, timedelta
from earthaccess import Auth
import logging

class Authenticator:
    
    def __init__(self, token_expiry_hours: float = 1.5, refresh_margin: timedelta = timedelta(minutes=10)):
        self.logger = logging.getLogger(__name__)
        self.auth = None
        self.token_expiry = None
        self.token_expiry_hours = token_expiry_hours
        self.refresh_margin = refresh_margin
    
    def login(self) -> bool:
        self.logger.info(" Auth on EarthData...")
        self.auth = Auth()
        self.auth.login(strategy="environment")
        
        if self.auth.authenticated:
            self.token_expiry = datetime.now() + timedelta(hours=self.token_expiry_hours)
            self.logger.info(f"Succesfully auth (valid at ~{self.token_expiry.strftime('%H:%M')})")
            return True
        else:
            self.logger.error("Error when authenticating")
            return False
    
    def should_refresh(self) -> bool:
        if self.token_expiry is None:
            return True
        return datetime.now() >= (self.token_expiry - self.refresh_margin)
    
    def refresh_if_needed(self):
        if self.should_refresh():
            self.logger.warning(" Refresh token...")
            self.login()
    
    def get_headers(self) -> dict:
        if not self.auth or not self.auth.authenticated:
            raise RuntimeError("Unauthenticated")
        return {"Authorization": f"Bearer {self.auth.token}"}