
try:
    import app
    print("app.py imported successfully")
except NameError as e:
    print(f"Caught expected NameError: {e}")
except Exception as e:
    print(f"Caught unexpected error: {type(e).__name__}: {e}")
