import streamlit as st


async def footer():
    footer = """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: transparent;
            color: #333;
            text-align: center;
            padding: 10px;
            box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.1);
            z-index: 100; /* Prevent overlaying of page content on footer */
        }
        </style>
        <div class="footer">
                    &copy; 2024. Made with 💖 <a href="https://www.linkedin.com/in/jyoti-dabass-ph-d-2b747083/" target="_blank" style="text-decoration: none;">Jyoti Dabass</a>
            <span style="color: #aaaaaa;">& Light ✨</span><br>
        </div>
        """

    return st.markdown(footer, unsafe_allow_html=True)
