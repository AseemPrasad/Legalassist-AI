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
    
    // Accessibility: Escape key listener
    this.escapeHandler = (e) => {
      if (e.key === 'Escape' && this.modal && this.modal.style.display === 'block') {
        this.hide();
      }
    };
  }

  show() {
    if (this.modal) {
      this.modal.style.display = 'block';
      this.modal.setAttribute('aria-hidden', 'false');
      this.modal.setAttribute('role', 'dialog');
      this.modal.setAttribute('aria-modal', 'true');
      document.addEventListener('keydown', this.escapeHandler);

      // Focus close button or first interactive element
      if (this.closeBtn) {
        this.closeBtn.focus();
      } else {
        const focusable = this.modal.querySelector('button, [href], input, select, textarea, [tabindex="0"]');
        if (focusable) focusable.focus();
      }
    }
  }

  hide() {
    if (this.modal) {
      this.modal.style.display = 'none';
      this.modal.setAttribute('aria-hidden', 'true');
      document.removeEventListener('keydown', this.escapeHandler);
    }
  }
}

export { CoreModal };
