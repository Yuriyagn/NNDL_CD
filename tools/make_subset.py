from __future__ import annotations

try:
    from tools.crop_levir import main
except ModuleNotFoundError:
    from crop_levir import main


if __name__ == "__main__":
    main()
