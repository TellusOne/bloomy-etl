"""
Gerenciamento de autenticação e token NASA EarthData
"""
from datetime import datetime, timedelta
from earthaccess import Auth
import logging

class Authenticator:
    """Gerencia autenticação e renovação de token"""
    
    def __init__(self, token_expiry_hours: float = 1.5, refresh_margin: timedelta = timedelta(minutes=10)):
        self.logger = logging.getLogger(__name__)
        self.auth = None
        self.token_expiry = None
        self.token_expiry_hours = token_expiry_hours
        self.refresh_margin = refresh_margin
    
    def login(self) -> bool:
        """Autentica no EarthData"""
        self.logger.info("🔐 Autenticando no EarthData...")
        self.auth = Auth()
        self.auth.login(strategy="environment")
        
        if self.auth.authenticated:
            self.token_expiry = datetime.now() + timedelta(hours=self.token_expiry_hours)
            self.logger.info(f"✅ Autenticado (válido até ~{self.token_expiry.strftime('%H:%M')})")
            return True
        else:
            self.logger.error("❌ Falha na autenticação")
            return False
    
    def should_refresh(self) -> bool:
        """Verifica se token precisa renovação"""
        if self.token_expiry is None:
            return True
        return datetime.now() >= (self.token_expiry - self.refresh_margin)
    
    def refresh_if_needed(self):
        """Renova token se necessário"""
        if self.should_refresh():
            self.logger.warning("🔄 Renovando token...")
            self.login()
    
    def get_headers(self) -> dict:
        """Retorna headers de autenticação para requests"""
        if not self.auth or not self.auth.authenticated:
            raise RuntimeError("Não autenticado")
        return {"Authorization": f"Bearer {self.auth.token}"}