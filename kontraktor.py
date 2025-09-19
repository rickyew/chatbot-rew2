import streamlit as st
import openai
import PyPDF2
import io
import tiktoken # Opsional, untuk menghitung token

# --- Fungsi untuk Mengekstrak Teks dari PDF ---
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# --- Fungsi untuk Berinteraksi dengan OpenAI API ---
def get_openai_response(api_key, prompt, text_content, model="gpt-3.5-turbo", max_tokens=1500):
    openai.api_key = api_key

    try:
        # Menggabungkan prompt pengguna dengan konten teks dokumen
        full_prompt = f"Analisis dokumen kontrak berikut ini:\n\n{text_content}\n\n{prompt}"

        # Opsional: Hitung token untuk menghindari error batas token
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(full_prompt))

        if num_tokens > 4000: # Contoh batas token, sesuaikan dengan model yang digunakan
            st.warning(f"Dokumen terlalu panjang ({num_tokens} token). Mungkin akan terpotong atau menghasilkan error.")
            # Jika terlalu panjang, Anda bisa memotong teks di sini
            # Atau meminta pengguna untuk ringkasan yang lebih singkat

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Anda adalah asisten AI yang ahli dalam menganalisis dokumen kontrak."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message['content']
    except openai.error.AuthenticationError:
        st.error("OpenAI API Key tidak valid. Mohon periksa kembali.")
        return None
    except openai.error.RateLimitError:
        st.error("Batas kuota OpenAI API terlampaui. Mohon tunggu sebentar atau upgrade paket Anda.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghubungi OpenAI API: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Penganalisis Dokumen Kontrak (PDF) dengan OpenAI", layout="wide")

st.title("📄 Penganalisis Dokumen Kontrak (PDF) dengan OpenAI")
st.write("Unggah dokumen kontrak PDF Anda dan gunakan OpenAI untuk menganalisisnya.")

# Sidebar untuk API Key
st.sidebar.header("Konfigurasi")
openai_api_key = st.sidebar.text_input("Masukkan OpenAI API Key Anda", type="password")
st.sidebar.info("Dapatkan API Key Anda dari [platform.openai.com](https://platform.openai.com/account/api-keys)")

# File Uploader
st.header("Unggah Dokumen PDF")
uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")

if uploaded_file is not None:
    st.success("File PDF berhasil diunggah!")

    # Membaca file PDF sebagai bytes
    pdf_bytes = uploaded_file.getvalue()

    # Mengekstrak teks dari PDF
    try:
        with io.BytesIO(pdf_bytes) as pdf_buffer:
            contract_text = extract_text_from_pdf(pdf_buffer)
        st.subheader("Pratinjau Teks yang Diekstrak (200 karakter pertama):")
        st.text(contract_text[:500] + "...")
        st.info(f"Jumlah karakter yang diekstrak: {len(contract_text)}")

        if not contract_text.strip():
            st.warning("Tidak dapat mengekstrak teks dari PDF. Pastikan PDF tidak berupa gambar.")
        else:
            st.header("Mulai Analisis")
            user_prompt = st.text_area(
                "Apa yang ingin Anda ketahui tentang kontrak ini?",
                "Berikan ringkasan poin-poin penting dari kontrak ini, siapa saja pihak yang terlibat, dan kewajiban utama masing-masing pihak."
            )

            if st.button("Analisis Dokumen"):
                if not openai_api_key:
                    st.warning("Mohon masukkan OpenAI API Key Anda di sidebar.")
                else:
                    with st.spinner("Menganalisis dokumen dengan OpenAI..."):
                        response_content = get_openai_response(openai_api_key, user_prompt, contract_text)
                        if response_content:
                            st.subheader("Hasil Analisis dari OpenAI:")
                            st.write(response_content)

    except PyPDF2.errors.PdfReadError:
        st.error("Gagal membaca file PDF. Pastikan ini adalah file PDF yang valid dan tidak terenkripsi.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses PDF: {e}")

else:
    st.info("Silakan unggah dokumen PDF untuk memulai analisis.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh asisten AI")

# Contoh tambahan: Gambar untuk visualisasi
st.sidebar.subheader("Visualisasi")
st.sidebar.write("Berikut adalah visualisasi kontrak")
