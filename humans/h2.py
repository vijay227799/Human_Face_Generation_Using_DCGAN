import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Mapping nationalities to their corresponding codes used by the API
nationality_codes = {
    "USA": "us",
    "UK": "gb",
    "France": "fr",
    "Denmark": "dk",
    "India": "in",  # Note: Random User API might not have data for some countries like India
    "Australia": "au",
    "Canada": "ca"
}

# Function to fetch a human face with specific attributes
def fetch_face_with_attributes(gender, nationality):
    # Convert to lowercase for API compatibility
    gender = gender.lower()
    nationality = nationality_codes.get(nationality, "").lower()

    # Construct the API request URL
    url = f"https://randomuser.me/api/?gender={gender}&nat={nationality}&inc=picture,gender,nat"
    
    response = requests.get(url).json()
    user_info = response['results'][0]

    # Ensure the returned user matches the requested gender and nationality
    if user_info['gender'] == gender and user_info['nat'].lower() == nationality:
        image_url = user_info['picture']['large']
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        return img
    else:
        # Retry fetching another user if attributes do not match
        return fetch_face_with_attributes(gender, nationality)

# Streamlit app
st.title("DCGAN Human Face Generator")

# Text input for user specifications
user_input = st.text_input("Enter your face specifications (e.g., 'Gender: Male, Nationality: USA'):")

if st.button("Generate Face"):
    try:
        # Extract gender and nationality from user input
        specs = [s.strip().lower() for s in user_input.split(",")]
        gender = next((spec.split(":")[1].strip().capitalize() for spec in specs if "gender" in spec), "Male")
        nationality = next((spec.split(":")[1].strip().capitalize() for spec in specs if "nationality" in spec), "USA")
        
        # Fetch the face with specified attributes
        image = fetch_face_with_attributes(gender, nationality)
        if image:
            st.image(image, caption='Generated Human Face', use_column_width=True)

            # Prepare image data for download
            img_bytes_png = BytesIO()
            img_bytes_jpg = BytesIO()
            image.save(img_bytes_png, format='PNG')
            image.save(img_bytes_jpg, format='JPEG')

            # Provide download buttons
            st.write("Download as:")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(label="Download as PNG", data=img_bytes_png.getvalue(),
                                   file_name="generated_face.png", mime="image/png")
            with col2:
                st.download_button(label="Download as JPG", data=img_bytes_jpg.getvalue(),
                                   file_name="generated_face.jpg", mime="image/jpeg")
        else:
            st.write("Could not fetch an image with the given specifications. Please try different attributes.")
    except Exception as e:
        st.write("Invalid input format or error fetching the image. Please enter specifications in the format: 'Gender: Male, Nationality: USA'")
