import streamlit as st
from PIL import Image

from classifier import classify


def draw_bar(item):
    key, value = list(item.items())[0]

    col1, col2 = st.columns([12, 1])

    col1.write(key.capitalize())
    col2.write("{:.4f}".format(value))

    st.progress(value)


def main():
    st.set_page_config(page_title="ConvNeXT", page_icon="🐝")
    st.title("🐝 ConvNeXT tiny classifier app")

    st.write("This is an image classification app based on ConvNeXT tiny-sized model.")
    st.write("Model source: https://huggingface.co/facebook/convnext-tiny-224")

    st.subheader("Choose a file")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )

    image_placeholder = st.container()
    button_placeholder = st.empty()
    results_placeholder = st.container()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_placeholder.subheader("Selected image")
        image_placeholder.image(image, use_column_width=True)

        button = button_placeholder.button("Classify", type="primary")

        if button:
            with button_placeholder:
                with st.spinner("Processing..."):
                    results = classify(image)
                    results_placeholder.subheader("Results")

                    tab1, tab2 = results_placeholder.tabs(["List", "Bar chart"])

                    with tab1:
                        for item in results:
                            draw_bar(item)

                    tab2.bar_chart(results)

    st.caption("by Vladislav Shalnev")


if __name__ == "__main__":
    main()
