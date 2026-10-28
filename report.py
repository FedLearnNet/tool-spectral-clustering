def html_report(title: str, params: dict, extra: dict) -> str:
    def _rows(d: dict) -> str:
        return "\n".join(
            f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"
            for k, v in d.items()
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 980px; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; text-align: left; }}
    code {{ background: #f1f1f1; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h2>{title}</h2>

  <h3>Parameters</h3>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    {_rows(params)}
  </table>

  <h3>Summary</h3>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    {_rows(extra)}
  </table>
</body>
</html>
"""
