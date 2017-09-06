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


;; System:melpa (Milkypostman’ Emacs Lisp Package Archive)
(require 'package)
(add-to-list
 'package-archives
 '("melpa" . "http://melpa.milkbox.net/packages/")
    t)

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
(setq x-select-enable-clipboard t)
(setq interprogram-paste-function 'x-cut-buffer-or-selection-value)

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

;; For python environment in Emacs,
;; install elpy from https://github.com/jorgenschaefer/elpy
;; e.g. pip install jedi flake8 importmagic autopep8
(package-initialize)
(elpy-enable)

;; py-autopep8
;; (require 'py-autopep8)
(add-hook 'python-mode-hook 'py-autopep8-enable-on-save)
```