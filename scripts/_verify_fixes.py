import sys

checks = {
    r'd:\Legal_Assist\Legalassist-AI\cli.py': [
        ('_reinitialize_semaphore', 'Semaphore helper defined'),
        ('Config.SUMMARY_MAX_TOKENS', 'Summary token limit config-driven'),
        ('Config.REMEDIES_MAX_TOKENS', 'Remedies token limit config-driven'),
        ('_reinitialize_semaphore(args.concurrency)', 'process_command/batch_command calls reinit'),
        ('from config import Config', 'Config imported in cli.py'),
        ('import hashlib', 'hashlib imported in cli.py'),
        # Checkpoint delete-before-load: the unlink must appear before load_checkpoint
    ],
    r'd:\Legal_Assist\Legalassist-AI\app.py': [
        ('import hashlib', 'hashlib imported in app.py'),
        ('content_hash = hashlib.md5', 'Content hash computed'),
        ('cache_key', 'Cache key variable used'),
        ('st.session_state.last_processed = cache_key', 'Cache key stored in session'),
    ],
    r'd:\Legal_Assist\Legalassist-AI\core\app_utils.py': [
        ('p_list = data.get("page_num", [])', 'OCR KeyError guarded'),
        ('Config.REMEDIES_MAX_TOKENS', 'Remedies token limit config-driven in app_utils'),
    ],
}

all_ok = True
for fpath, items in checks.items():
    src = open(fpath, encoding='utf-8').read()
    for needle, label in items:
        found = needle in src
        status = 'OK' if found else 'MISSING'
        if not found:
            all_ok = False
        print(f'[{status}] {label}')

# Extra check: in cli.py, unlink() must appear before load_checkpoint
cli_src = open(r'd:\Legal_Assist\Legalassist-AI\cli.py', encoding='utf-8').read()
unlink_pos = cli_src.find('checkpoint_file.unlink()')
load_pos = cli_src.find('load_checkpoint(checkpoint_file)')
if unlink_pos != -1 and load_pos != -1 and unlink_pos < load_pos:
    print('[OK] Checkpoint deleted BEFORE load_checkpoint (ordering fix)')
else:
    print('[MISSING] Checkpoint delete-before-load ordering fix')
    all_ok = False

# Extra check: no st.spinner wrapping remedies display in app.py
app_src = open(r'd:\Legal_Assist\Legalassist-AI\app.py', encoding='utf-8').read()
if 'remedies_spinner' not in app_src:
    print('[OK] Misleading remedies spinner removed')
else:
    print('[MISSING] Misleading remedies spinner still present')
    all_ok = False

sys.exit(0 if all_ok else 1)
