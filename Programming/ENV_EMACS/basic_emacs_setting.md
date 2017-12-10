# Basic EMACS setting
- This EMACS setting helps you...
	- show line number
	- set line formatting
	- enable copy to clipboard
	- set color themes
	- set useful python environment

## General Settings
```shell
;;;;;;;;;;;;;;;;;;;;;;;;
;;; General Settings ;;;
;;;;;;;;;;;;;;;;;;;;;;;;


(require 'package)
(add-to-list 'package-archives
             '("melpa-stable" . "https://stable.melpa.org/packages/"))
(package-initialize)
(when (not package-archive-contents)
    (package-refresh-contents))

;; Display:see lines
(global-linum-mode 1)
(setq linum-format "%d ")
(add-hook 'find-file-hook (lambda () (linum-mode 1)))

;; Display:line delimiting
(unless window-system
  (add-hook 'linum-before-numbering-hook
	    (lambda ()
	      (setq-local linum-format-fmt
			  (let ((w (length (number-to-string
					    (count-lines (point-min) (point-max))))))
			    (concat "%" (number-to-string w) "d"))))))

;; Display: show line number
(defun linum-format-func (line)
  (concat
   (propertize (format linum-format-fmt line) 'face 'linum)
   (propertize " " 'face 'mode-line)))
(unless window-system
      (setq linum-format 'linum-format-func))

;; System: copy to clipboard
;; (setq x-select-enable-clipboard t)
;; (setq interprogram-paste-function 'x-cut-buffer-or-selection-value)
(setq interprogram-cut-function
      (lambda (text &optional push)
    (let* ((process-connection-type nil)
           (pbproxy (start-process "pbcopy" "pbcopy" "/usr/bin/pbcopy")))
      (process-send-string pbproxy text)
      (process-send-eof pbproxy))))


;; System: enable mouse support
(unless window-system
  (require 'mouse)
  ;; (xterm-mouse-mode t)
  (global-set-key [mouse-4] (lambda ()
			      (interactive)
			      (scroll-down 1)))
  (global-set-key [mouse-5] (lambda ()
			      (interactive)
			      (scroll-up 1)))
  (defun track-mouse (e))
  (setq mouse-sel-mode t)
  )


;; System: use Korean
;; (set-language-environment "Korean")
;; (prefer-coding-system 'utf-8)

;; System: replace the selected text with the contents of the clipboard
(delete-selection-mode)
```

## Themes
```shell
;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;; Themes ;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;


;; My custom theme setting
(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(ansi-color-faces-vector
   [default default default italic underline success warning error])
 '(ansi-color-names-vector
   ["#212526" "#ff4b4b" "#b4fa70" "#fce94f" "#729fcf" "#e090d7" "#8cc4ff" "#eeeeec"])
 '(custom-enabled-themes (quote (tango-dark)))
 '(custom-safe-themes
   (quote
    ("dbb643699e18b5691a8baff34c29d709a3ff9787f09cdae58d3c1bc085b63c25" default))))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )
```

## Python Settings
```shell
;;;;;;;;;;;;;;;;;;;;;;;;
;;; Python Settings ;;;;
;;;;;;;;;;;;;;;;;;;;;;;;

(defvar myPackages
  '(better-defaults
    ein
    elpy
    flycheck
    material-theme
    py-autopep8)) ;; add the autopep8 package
(elpy-enable)

;; Syntax checking
(when (require 'flycheck nil t)
  (setq elpy-modules (delq 'elpy-module-flymake elpy-modules))
  (add-hook 'elpy-mode-hook 'flycheck-mode))

;; py-autopep8
(require 'py-autopep8)
(add-hook 'elpy-mode-hook 'py-autopep8-enable-on-save)
```