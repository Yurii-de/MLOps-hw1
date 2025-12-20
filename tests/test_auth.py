from datetime import timedelta
from src.auth.jwt_handler import create_access_token, get_password_hash, verify_password, decode_access_token

def test_password_hashing():
    password = "secret_password"
    hashed = get_password_hash(password)
    
    assert verify_password(password, hashed)
    assert not verify_password("wrong_password", hashed)
    assert hashed != password

def test_jwt_token_creation_and_decoding():
    data = {"sub": "test_user"}
    token = create_access_token(data, expires_delta=timedelta(minutes=15))
    
    decoded = decode_access_token(token)
    
    assert decoded is not None
    assert decoded["sub"] == "test_user"
    # Check that expiration is set (approximate check if needed, but presence is enough for now)
    assert "exp" in decoded

def test_jwt_token_expiration():
    # Create a token that expires immediately
    data = {"sub": "test_user"}
    token = create_access_token(data, expires_delta=timedelta(seconds=-1))
    
    # Decoding should return None for expired token (assuming decode_access_token handles it)
    # Let's check decode_access_token implementation first to be sure.
    decoded = decode_access_token(token)
    assert decoded is None
