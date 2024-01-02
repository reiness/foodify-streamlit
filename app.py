import streamlit as st
from PIL import Image
import io
from ultralytics import YOLO
from openai import OpenAI
import time


def convert_to_jpg(image):
    # Force resize the image to 640x640
    resized_image = image.resize((640, 640))

    # Konversi gambar ke format JPG
    with io.BytesIO() as output:
        resized_image.save(output, format="JPEG")
        jpg_image = Image.open(io.BytesIO(output.getvalue()))

    return jpg_image


def main():
    st.title("Foodify")
    model = YOLO('models/640best-s.pt')  

    # Menentukan lebar kolom
    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("Masukkan foto bahan masakmu disini", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Baca gambar yang diunggah
            image = Image.open(uploaded_file)

            # Konversi ke format JPG
            jpg_image = convert_to_jpg(image)
            results = model.predict(jpg_image,stream=True, show=True, save=False, imgsz=640) # source already setup
            bahan_classes = model.names

            bahan = "Unknown"  # Default value in case index is out of range

        for r in results:
            for c in r.boxes.cls:
                c_int = int(c)
                if 0 <= c_int < len(bahan_classes):
                    bahan = bahan_classes[c_int]
            st.image(jpg_image, caption=f"{bahan}", use_column_width=True)
        
    # st.write(bahan)

    with col2:
        if uploaded_file is not None:
            prompt = f"""
            Kamu adalah seorang Chef handal. Kamu bisa memasak apa saja berdasarkan bahan masakan yang ada. 
            Dalam kasus kali ini, client mu adalah seorang mahasiswa kos yang ingin memasak berdasarkan bahan yang ada namun tidak tahu bahan tersebut bisa digunakan untuk masak apa.

            Bahan masakan yang dimiliki anak kos ini adalah: {bahan}

            Based only AND ONLY from the ingredients above, buatlah sebuah resep yang cocok untuk mahasiswa ini beserta langkah-langkah cara memasaknya !
            Ingatlah untuk tidak memasukkan bahan-bahan selain yang dimiliki oleh client!
            """
        
            client = OpenAI(
            api_key=st.secrets['OPENAI_API_KEY'],
            )

            Foodify = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo",
            )


            if bahan:
                with st.spinner("Foodify sedang memproses bahan masakanmu"):
                    time.sleep(2)  # Simulate a time-consuming operation
                    if Foodify.choices:
                        out = Foodify.choices[0].message.content
                    else:
                        out = None
                    # Delay for demonstration purposes
                    
                st.success(out if out is not None else '')


if __name__ == "__main__":
    main()
