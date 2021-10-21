import streamlit as st
import os
from nemo.collections.nlp.models import MTEncDecModel
MTEncDecModel.list_available_models()

def main():
	""" NLP Based App with Streamlit """

	# Title
	st.title("NeMo webapp")
	st.subheader("Neural Machine Translation (NMT) model to translate from One language to Another")
	


	if st.checkbox("translate from English to Hindi"):
		

		message1 = st.text_area("Enter Text","Type Here....")
		model1 = MTEncDecModel.from_pretrained("nmt_en_hi_transformer12x2")
		translations1 = model1.translate([message1], source_lang="en", target_lang="hi")
		st.success(translations1[0])

	elif st.checkbox("translate from English to spanish"):
		

		message2 = st.text_area("Enter Text","Type Here....")
		model2 = MTEncDecModel.from_pretrained("nmt_en_es_transformer12x2")
		translations2 = model2.translate([message2], source_lang="en", target_lang="es")
		st.success(translations2[0])

	elif st.checkbox("translate from English to german"):
		

		message3 = st.text_area("Enter Text","Type Here....")
		model3 = MTEncDecModel.from_pretrained("nmt_en_de_transformer12x2")
		translations3 = model3.translate([message3], source_lang="en", target_lang="de")
		st.success(translations3[0])






	st.sidebar.subheader("About the App")
	st.sidebar.text("NeMo by NVIDIA (nmt inference tool")
	st.sidebar.info("Use this tool to get the translation from one language to other ")
	st.sidebar.subheader("Developed by")
	st.sidebar.text("Somil Jain")




if __name__ == '__main__':
	main()
