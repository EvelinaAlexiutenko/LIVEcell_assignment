import os
from threading import Thread
from pyngrok import ngrok


def run_streamlit():
    os.system("streamlit run ./app.py --server.port 8501")


# Start the Streamlit app in a separate thread
thread = Thread(target=run_streamlit)
thread.start()
# Create a public URL for the Streamlit app using ngrok
public_url = ngrok.connect(
    addr="8501", proto="http", bind_tls=True
)  # if you do need a public access 
print("Your Streamlit app is live at:", public_url)
