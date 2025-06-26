# irr-dashboard
IRR Analysis Dashboard with Streamlit


All proprietary investment information (DST names, equity, etc) is hidden server-side.

To configure this information, include the following in your `secrets.toml` file:
```
[auth]
password = "password123"

[dst1]
name   = "DST1"
equity = 42 # initial $

[dst1.perc] # annual dist. %
1  = 0.42
2 = 0.042
3 = 4.2

[secret_text]
text = """
Put your info here.
This text block is in paragraph form.
Format colors, line spacing, etc in the code.
"""
```
