"""
window.py - the live window the classifier scripts draw into.

Shared the same way report.py is. Any script that classifies one stamp at a
time can open a window and push a frame per stamp, so you watch the run
happen instead of reading it scroll past.

    from window import Window
    win = Window("CLIP zero-shot", "four sentences, nothing trained", CLASSES)
    ...
    if not win.update(path, truth, scores=probs, secs=elapsed):
        break            # the user pressed q
    ...
    win.close()

Two shapes of result are supported. Pass `scores` for a model that produces a
number per bin, and the window draws bars. Pass `caption` for a model that
answers in words, and it draws the sentence and which keywords matched.

    space  pause / resume        q or esc  stop the run
"""
import os
import time

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from classes import REJECT_THRESHOLD

W, H = 1440, 820

BG     = (18, 22, 27)
PANEL  = (26, 32, 40)
TRACK  = (35, 43, 52)
TEXT   = (230, 235, 239)
MUTED  = (138, 153, 168)
FAINT  = (80, 92, 105)
GOOD   = (110, 200, 150)
REJECT = (217, 105, 90)
BIN_COLOR = {"nature": (76, 175, 125), "people": (91, 141, 239),
             "places": (224, 163, 62), "graphic": (155, 127, 224)}
FALLBACK = (150, 160, 172)

_FONT_DIRS = ["/System/Library/Fonts/Supplemental/", "/System/Library/Fonts/",
              "/Library/Fonts/", "C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/"]
_SANS = ["HelveticaNeue.ttc", "Helvetica.ttc", "Arial.ttf", "arial.ttf", "DejaVuSans.ttf"]
_MONO = ["Menlo.ttc", "Courier New.ttf", "cour.ttf", "DejaVuSansMono.ttf"]


def _font(names, size):
    for d in _FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


F_HUGE, F_BIG = _font(_SANS, 40), _font(_SANS, 27)
F_MED, F_SMALL = _font(_SANS, 22), _font(_SANS, 16)
F_CAPTION = _font(_SANS, 26)
F_TINY, F_LABEL = _font(_MONO, 14), _font(_MONO, 13)


def _rr(d, box, r, **kw):
    d.rounded_rectangle(box, radius=r, **kw)


def _wrap(d, text, font, width):
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if d.textlength(trial, font=font) <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = word
    if cur:
        lines.append(cur)
    return lines


class Window:
    def __init__(self, title, subtitle, classes, threshold=REJECT_THRESHOLD,
                 name="stamp sorter", pace=0.0):
        self.title, self.subtitle = title, subtitle
        self.classes, self.threshold, self.name = classes, threshold, name
        self.pace = pace   # seconds to hold each stamp, for replays
        self.total = self.correct = 0
        self.times = []
        self.paused = False
        self.open = True
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    # ------------------------------------------------------------------ draw
    def _frame(self, path, truth, scores, caption, matched, pred, secs, shown, options):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)

        rejected = scores is not None and float(np.max(scores)) < self.threshold
        if rejected:
            verdict, vcol = "reject bin", REJECT
        elif pred in self.classes:
            verdict, vcol = pred, BIN_COLOR.get(pred, FALLBACK)
        else:
            verdict, vcol = "no match", REJECT
        correct = (not rejected) and pred == truth

        d.text((48, 40), "STAMP SORTER  \u00b7  RUNNING", font=F_LABEL, fill=FAINT)
        run = "%d / %d correct   %.0f%%" % (self.correct, self.total,
                                            100.0 * self.correct / max(1, self.total))
        d.text((W - 48 - d.textlength(run, font=F_MED), 34), run, font=F_MED, fill=MUTED)

        # ---------------- left: the stamp
        _rr(d, (48, 84, 600, 772), 14, fill=PANEL)
        try:
            t = Image.open(path).convert("RGB")
        except Exception:
            t = Image.new("RGB", (300, 380), (40, 40, 40))
        t.thumbnail((470, 540))
        tx, ty = 48 + (552 - t.width) // 2, 84 + (600 - t.height) // 2 + 20
        d.rectangle((tx - 3, ty - 3, tx + t.width + 2, ty + t.height + 2), fill=(55, 65, 76))
        im.paste(t, (tx, ty))

        d.text((80, 712), os.path.basename(path), font=F_TINY, fill=FAINT)
        d.text((80, 736), "sorted by hand as", font=F_SMALL, fill=MUTED)
        off = d.textlength("sorted by hand as", font=F_SMALL) + 10
        d.text((80 + off, 734), str(truth), font=F_MED,
               fill=BIN_COLOR.get(truth, FALLBACK))

        # ---------------- right
        X0, X1 = 648, W - 48
        d.text((X0, 92), "SCORING METHOD", font=F_LABEL, fill=FAINT)
        d.text((X0, 116), self.title, font=F_BIG, fill=TEXT)
        d.text((X0, 152), self.subtitle, font=F_SMALL, fill=MUTED)

        if scores is not None:
            bx0, bx1 = X0 + 132, X1 - 92
            top, step, bh = 224, 84, 34
            win_i = int(np.argmax(scores))
            for i, c in enumerate(self.classes):
                yy = top + i * step
                is_win, col = i == win_i, BIN_COLOR.get(c, FALLBACK)
                dim = tuple(int(v * 0.42) for v in col)
                d.text((X0, yy + 6), c, font=F_MED if is_win else F_SMALL,
                       fill=TEXT if is_win else MUTED)
                _rr(d, (bx0, yy, bx1, yy + bh), bh // 2, fill=TRACK)
                if shown[i] > 0.004:
                    wpx = int((bx1 - bx0) * float(shown[i]))
                    _rr(d, (bx0, yy, bx0 + max(bh, wpx), yy + bh), bh // 2,
                        fill=col if is_win else dim)
                val = "%.0f%%" % (scores[i] * 100)
                f = F_MED if is_win else F_SMALL
                d.text((X1 - d.textlength(val, font=f), yy + 5), val, font=f,
                       fill=TEXT if is_win else MUTED)
            tx2 = bx0 + (bx1 - bx0) * self.threshold
            for yy in range(top - 14, top + 3 * step + bh + 14, 9):
                d.line((tx2, yy, tx2, yy + 5), fill=FAINT, width=2)
            lab = "reject below %.2f" % self.threshold
            d.text((tx2 - d.textlength(lab, font=F_LABEL) / 2,
                    top + 3 * step + bh + 20), lab, font=F_LABEL, fill=FAINT)
        else:
            d.text((X0, 214), "IT ANSWERED", font=F_LABEL, fill=FAINT)
            _rr(d, (X0, 238, X1, 372), 12, fill=PANEL)
            lines = _wrap(d, '"%s"' % (caption or ""), F_CAPTION, X1 - X0 - 56)[:3]
            for i, ln in enumerate(lines):
                d.text((X0 + 28, 262 + i * 36), ln, font=F_CAPTION, fill=TEXT)
            if matched is not None:
                d.text((X0, 404), "KEYWORDS MATCHED", font=F_LABEL, fill=FAINT)
                if matched:
                    cx = X0
                    for word in matched[:6]:
                        wpx = d.textlength(word, font=F_SMALL) + 28
                        if cx + wpx > X1:
                            break
                        col = BIN_COLOR.get(pred, FALLBACK)
                        _rr(d, (cx, 430, cx + wpx, 466), 18, fill=PANEL,
                            outline=col, width=1)
                        d.text((cx + 14, 438), word, font=F_SMALL, fill=col)
                        cx += wpx + 10
                else:
                    d.text((X0, 434), "none \u2014 nothing in our word list fits this",
                           font=F_SMALL, fill=REJECT)
            elif options:
                d.text((X0, 404), "IT WAS ALLOWED THESE FOUR WORDS", font=F_LABEL, fill=FAINT)
                cx = X0
                for word in options:
                    wpx = d.textlength(word, font=F_SMALL) + 28
                    chosen = word == pred
                    col = BIN_COLOR.get(word, FALLBACK) if chosen else FAINT
                    _rr(d, (cx, 430, cx + wpx, 466), 18,
                        fill=PANEL if chosen else BG, outline=col, width=2 if chosen else 1)
                    d.text((cx + 14, 438), word, font=F_SMALL,
                           fill=col if chosen else MUTED)
                    cx += wpx + 12

        # ---------------- verdict
        vy = 648
        _rr(d, (X0, vy, X1, vy + 92), 12, fill=PANEL)
        _rr(d, (X0, vy, X0 + 6, vy + 92), 3, fill=vcol)
        d.text((X0 + 28, vy + 18), "GOES TO", font=F_LABEL, fill=FAINT)
        d.text((X0 + 28, vy + 38), verdict, font=F_HUGE, fill=vcol)
        mark = "correct" if correct else ("held back" if rejected else "wrong")
        mcol = GOOD if correct else (MUTED if rejected else REJECT)
        d.text((X1 - 28 - d.textlength(mark, font=F_MED), vy + 50), mark,
               font=F_MED, fill=mcol)

        foot = "space  pause        q  stop the run"
        d.text((48, H - 34), foot, font=F_SMALL, fill=FAINT)
        if self.times:
            pace = "%.2fs per stamp" % (sum(self.times) / len(self.times))
            d.text((W - 48 - d.textlength(pace, font=F_SMALL), H - 34), pace,
                   font=F_SMALL, fill=FAINT)
        if self.paused:
            d.text((W // 2 - 40, H - 34), "PAUSED", font=F_SMALL, fill=REJECT)

        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    # ---------------------------------------------------------------- update
    def update(self, path, truth, scores=None, caption=None, matched=None,
               pred=None, secs=None, frames=7, options=None):
        """Draw one stamp. Returns False if the user asked to stop."""
        if not self.open:
            return False
        if scores is not None:
            scores = np.asarray(scores, dtype=float)
            if pred is None:
                pred = self.classes[int(scores.argmax())]
        self.total += 1
        if pred == truth and not (scores is not None and float(scores.max()) < self.threshold):
            self.correct += 1
        if secs:
            self.times.append(secs)

        started = time.time()
        shown = np.zeros(len(self.classes))
        for _ in range(max(1, frames)):
            if scores is not None:
                shown += (scores - shown) * 0.42
            cv2.imshow(self.name, self._frame(path, truth, scores, caption,
                                              matched, pred, secs, shown, options))
            if not self._keys(1):
                return False
        while self.paused:
            if not self._keys(60):
                return False
        while time.time() - started < self.pace:
            if not self._keys(20):
                return False
            while self.paused:
                if not self._keys(60):
                    return False
        return True

    def _keys(self, wait):
        k = cv2.waitKey(wait) & 0xFF
        if k in (27, ord("q")):
            self.open = False
            cv2.destroyAllWindows()
            return False
        if k == 32:
            self.paused = not self.paused
        return True

    def close(self, hold=True):
        if self.open and hold:
            cv2.waitKey(1200)
        self.open = False
        cv2.destroyAllWindows()
