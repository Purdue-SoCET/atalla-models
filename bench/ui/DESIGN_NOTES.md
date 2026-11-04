# Workbench design notes (from ~/Documents/numa, Robert's desktop app)

Restyle target for `bench/ui/static`. Numa is Apple-native in feel: quiet, hairline, light by default, dark by `prefers-color-scheme`. Keep the workbench's structure (rail, centre, context column) but adopt these tokens and rules.

## Tokens (copy verbatim into `:root` / dark block)

Light:
```
--bg:#ffffff; --side:rgba(246,246,245,.72); --side-solid:#f6f6f5; --raised:#ffffff; --pop:#ffffff; --fill:#f4f4f5;
--text:#1d1d1f; --text-2: rgba(29,29,31,.62); --text-3: rgba(29,29,31,.42);
--line:rgba(0,0,0,.075); --hover:rgba(0,0,0,.042); --active:rgba(0,0,0,.072);
--accent:#0a7cff; --accent-soft:rgba(10,124,255,.11); --primary:#1d1d1f; --on-primary:#ffffff;
--green:#1c9a4f; --red:#e0383e; --amber:#b9770e;
--shadow-pop:0 0 0 .5px rgba(0,0,0,.12),0 12px 32px rgba(0,0,0,.12),0 2px 6px rgba(0,0,0,.05);
--shadow-composer:0 0 0 .5px rgba(0,0,0,.11),0 2px 10px rgba(0,0,0,.04);
--scrim:rgba(0,0,0,.22);
```
Dark (`@media (prefers-color-scheme: dark)` and `[data-theme=dark]`):
```
--bg:#1b1b1c; --side:rgba(26,26,28,.82); --side-solid:#151516; --raised:#212123; --pop:#262628; --fill:#262628;
--text:#ececee; --text-2: rgba(236,236,238,.62); --text-3: rgba(236,236,238,.42);
--line:rgba(255,255,255,.075); --hover:rgba(255,255,255,.05); --active:rgba(255,255,255,.085);
--accent:#3b95ff; --accent-soft:rgba(59,149,255,.18); --primary:#ececee; --on-primary:#1b1b1c;
--green:#3fbf73; --red:#ff6166; --amber:#e3a33a;
--shadow-pop:0 0 0 .5px rgba(255,255,255,.1),0 14px 36px rgba(0,0,0,.5);
--shadow-composer:0 0 0 .5px rgba(255,255,255,.1),0 2px 10px rgba(0,0,0,.25);
--scrim:rgba(0,0,0,.45);
```
Fonts:
```
--sans: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif;
--display: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI Variable Display', 'Segoe UI', system-ui, sans-serif;
--mono: 'SF Mono', ui-monospace, 'Cascadia Code', Menlo, Consolas, monospace;
```
`body { font: 13px/1.45 var(--sans); font-feature-settings:'cv11','ss01'; letter-spacing:-.003em; -webkit-font-smoothing:antialiased; }`

## Rules Numa follows

- Sidebar: `background: var(--side)` with backdrop blur, 8px horizontal padding; nav items 30px tall, 7px radius, 500 weight, `--hover`/`--active` backgrounds, no borders. Section heads 12.5px 600 in `--text-2`.
- Icon buttons 28px square, 7px radius, no border, `--text-2`, hover `--hover`. Tiny variant 22px/5px.
- Cards/"raised" surfaces: `--raised` + `--shadow-composer` (NOT a 1px border), radius 20px for composer-like panels, 10px for code/pre, 18px for bubbles, 9px for inline alerts.
- Hairlines only via `--line`; never solid #ccc borders.
- Headings: `--display`, 600, tight letter-spacing (-0.022em at 30px, -0.01em in prose).
- Code: `--mono` 12.5px in `--fill` with 10px radius; inline code 11.5-12.5px with 5px radius.
- Error: `color: var(--red); background: color-mix(in srgb, var(--red) 7%, transparent)`.
- Focus ring: `outline: 2px solid var(--accent); outline-offset: 1px; border-radius: 6px`.
- Scrollbars invisible until hover (transparent thumb with padding-box trick).
- Streaming indicator: 7px pulsing dot after the last element.
- Status colours: green/amber/red tokens above; use `--accent-soft` for selected rows and chips.
- Density: 8px grid, 13px base, generous whitespace, few lines. Light mode is the default presentation; dark must be equally finished.

## Workbench-specific mapping

- Backend colours on the graph: emulator → `--accent`; numpy → `--amber`; passthrough → `--text-3`; skip → `--red`; graph (placeholder/get_attr/output) → `--line` outline, `--fill` body.
- Cosine badge: ≥0.999 `--green`, ≥0.99 `--amber`, else `--red`, drawn as a soft chip (`color-mix(in srgb, var(--green) 12%, transparent)` background).
- Feed cards (run progress) are quiet rows (hairline separators), not boxed tweets.
- Primary "Run" button: `--primary` background, `--on-primary` text, 8px/14px padding, 999px radius.
