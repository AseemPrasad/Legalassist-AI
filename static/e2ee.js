/**
 * End-to-End Encryption for LegalAssist AI
 *
 * Client-side encryption using WebCrypto API.
 * The server never sees plaintext document contents.
 *
 * Key hierarchy:
 *   User Passphrase -> PBKDF2 -> Master Key
 *   Master Key + File Salt -> PBKDF2 -> File Key
 *   File Key -> AES-256-GCM -> Encrypted File
 *
 * Usage:
 *   const key = await deriveKey(passphrase, salt);
 *   const encrypted = await encryptFile(fileBuffer, key);
 *   const decrypted = await decryptFile(encrypted, key);
 */

const PBKDF2_ITERATIONS = 600000;
const KEY_SIZE = 256;
const SALT_LEN = 32;
const IV_LEN = 12;

/**
 * Convert ArrayBuffer to base64 string
 */
function bufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Convert base64 string to ArrayBuffer
 */
function base64ToBuffer(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

/**
 * Generate a random salt for key derivation
 */
function generateSalt() {
  return crypto.getRandomValues(new Uint8Array(SALT_LEN));
}

/**
 * Generate a random IV for AES-GCM
 */
function generateIV() {
  return crypto.getRandomValues(new Uint8Array(IV_LEN));
}

/**
 * Derive a 256-bit AES key from a passphrase and salt using PBKDF2
 */
async function deriveKey(passphrase, salt) {
  const encoder = new TextEncoder();
  const passphraseKey = await crypto.subtle.importKey(
    'raw',
    encoder.encode(passphrase),
    'PBKDF2',
    false,
    ['deriveKey']
  );
  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: salt,
      iterations: PBKDF2_ITERATIONS,
      hash: 'SHA-256',
    },
    passphraseKey,
    { name: 'AES-GCM', length: KEY_SIZE },
    false,
    ['encrypt', 'decrypt']
  );
}

/**
 * Encrypt a file buffer using AES-256-GCM
 * @param {ArrayBuffer} plaintext - The file content to encrypt
 * @param {CryptoKey} key - AES-256-GCM key
 * @returns {{ciphertext: string, iv: string, salt: string}} base64-encoded payload
 */
async function encryptBuffer(plaintext, key) {
  const iv = generateIV();
  const ciphertext = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv: iv },
    key,
    plaintext
  );
  return {
    ct: bufferToBase64(ciphertext),
    iv: bufferToBase64(iv),
  };
}

/**
 * Decrypt a ciphertext using AES-256-GCM
 * @param {{ct: string, iv: string}} payload - Base64-encoded ciphertext and IV
 * @param {CryptoKey} key - AES-256-GCM key
 * @returns {ArrayBuffer} Decrypted plaintext
 */
async function decryptBuffer(payload, key) {
  const ciphertext = base64ToBuffer(payload.ct);
  const iv = base64ToBuffer(payload.iv);
  return crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: iv },
    key,
    ciphertext
  );
}

/**
 * Encrypt a file with passphrase-based key derivation
 * @param {File} file - File object from file input
 * @param {string} passphrase - User passphrase
 * @returns {Promise<{ciphertext: string, iv: string, salt: string}>} Encrypted payload (salt included separately)
 */
async function encryptFile(file, passphrase) {
  const salt = generateSalt();
  const key = await deriveKey(passphrase, salt);
  const plaintext = await file.arrayBuffer();
  const payload = await encryptBuffer(plaintext, key);
  return { ...payload, s: bufferToBase64(salt), v: 1 };
}

/**
 * Decrypt a file with passphrase-based key derivation
 * @param {{ct: string, iv: string, s: string, v: number}} payload - Encrypted payload
 * @param {string} passphrase - User passphrase
 * @returns {Promise<ArrayBuffer>} Decrypted file content
 */
async function decryptFile(payload, passphrase) {
  const salt = base64ToBuffer(payload.s);
  const key = await deriveKey(passphrase, salt);
  return decryptBuffer(payload, key);
}

/**
 * Encrypt a raw ArrayBuffer with passphrase
 * @param {ArrayBuffer} plaintext
 * @param {string} passphrase
 * @returns {Promise<{ct: string, iv: string, s: string, v: number}>}
 */
async function encryptBufferWithPassphrase(plaintext, passphrase) {
  const salt = generateSalt();
  const key = await deriveKey(passphrase, salt);
  const payload = await encryptBuffer(plaintext, key);
  return { ...payload, s: bufferToBase64(salt), v: 1 };
}

/**
 * Decrypt a raw ArrayBuffer with passphrase
 * @param {{ct: string, iv: string, s: string, v: number}} payload
 * @param {string} passphrase
 * @returns {Promise<ArrayBuffer>}
 */
async function decryptBufferWithPassphrase(payload, passphrase) {
  const salt = base64ToBuffer(payload.s);
  const key = await deriveKey(passphrase, salt);
  return decryptBuffer(payload, key);
}

/**
 * Generate a random file key (for envelope encryption)
 */
function generateFileKey() {
  return bufferToBase64(crypto.getRandomValues(new Uint8Array(32)));
}

/**
 * Wrap (encrypt) a file key with the user's master key
 */
async function wrapFileKey(fileKey, masterPassphrase, salt) {
  const key = await deriveKey(masterPassphrase, salt);
  const iv = generateIV();
  const wrapped = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv: iv },
    key,
    new TextEncoder().encode(fileKey)
  );
  const combinedSalt = generateSalt();
  const combinedKey = await deriveKey(masterPassphrase, combinedSalt);
  const combinedIV = generateIV();
  const outer = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv: combinedIV },
    combinedKey,
    new Uint8Array(wrapped)
  );
  return bufferToBase64(combinedSalt) + '.' +
         bufferToBase64(combinedIV) + '.' +
         bufferToBase64(outer);
}

/**
 * Unwrap (decrypt) a wrapped file key
 */
async function unwrapFileKey(wrapped, masterPassphrase) {
  const parts = wrapped.split('.');
  if (parts.length !== 3) throw new Error('Invalid wrapped key format');
  const [saltB64, ivB64, outerB64] = parts;
  const salt = base64ToBuffer(saltB64);
  const iv = base64ToBuffer(ivB64);
  const outer = base64ToBuffer(outerB64);
  const key = await deriveKey(masterPassphrase, salt);
  const fileKeyBuffer = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: iv },
    key,
    outer
  );
  return new TextDecoder().decode(fileKeyBuffer);
}

/**
 * Encrypt file bytes using a direct file key (envelope encryption)
 */
async function encryptWithFileKey(plaintext, fileKeyB64) {
  const keyData = base64ToBuffer(fileKeyB64);
  const key = await crypto.subtle.importKey(
    'raw', keyData,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
  const iv = generateIV();
  const ciphertext = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv: iv },
    key,
    plaintext
  );
  return { ct: bufferToBase64(ciphertext), iv: bufferToBase64(iv), v: 1 };
}

/**
 * Decrypt file bytes using a direct file key (envelope encryption)
 */
async function decryptWithFileKey(payload, fileKeyB64) {
  const keyData = base64ToBuffer(fileKeyB64);
  const key = await crypto.subtle.importKey(
    'raw', keyData,
    { name: 'AES-GCM', length: 256 },
    false,
    ['decrypt']
  );
  return crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: base64ToBuffer(payload.iv) },
    key,
    base64ToBuffer(payload.ct)
  );
}

export {
  encryptFile,
  decryptFile,
  encryptBufferWithPassphrase,
  decryptBufferWithPassphrase,
  deriveKey,
  generateSalt,
  generateFileKey,
  wrapFileKey,
  unwrapFileKey,
  encryptWithFileKey,
  decryptWithFileKey,
  bufferToBase64,
  base64ToBuffer,
};