/**
 * Reusable modal component utility.
 */
class CoreModal {
  constructor(elementId) {
    this.modal = document.getElementById(elementId);
    this.closeBtn = this.modal ? this.modal.querySelector('.close-modal') : null;
    if (this.closeBtn) {
      this.closeBtn.addEventListener('click', () => this.hide());
    }
  }

  show() {
    if (this.modal) {
      this.modal.style.display = 'block';
      this.modal.setAttribute('aria-hidden', 'false');
    }
  }

  hide() {
    if (this.modal) {
      this.modal.style.display = 'none';
      this.modal.setAttribute('aria-hidden', 'true');
    }
  }
}

export { CoreModal };
