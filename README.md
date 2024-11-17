# AI Migration Assistant - MVP for Portugal

## Overview
The **AI Migration Assistant** is an innovative application designed to simplify the migration process. Currently, as an MVP tailored for Portugal, it aims to provide centralized access to essential resources, making the journey to a new country less overwhelming. The future vision of the platform is to scale globally, assisting migrants with comprehensive tools for settling into any country.

---

## Features
### 1. **Housing Recommendations**
   - Personalized housing suggestions based on the user’s budget and preferences.
   - Integration with the Idealista API to fetch rental properties.
   - Dynamic recommendations formatted for easy browsing.

### 2. **Visa and Document Assistant**
   - AI-powered chatbot capable of answering visa-related queries.
   - Provides insights into processes like applying for a NIF, NISS, and residency permits.
   - Handles fallback responses via GPT for unmatched queries.

### 3. **Public Office and Embassy Directory**
   - Comprehensive directory of:
     - Finance offices
     - Social security offices
     - AIIMA offices
     - Embassies in Portugal
     - Portuguese embassies abroad
   - JSON-powered data for seamless integration.

### 4. **Cost of Living and City Insights**
   - Categorization of cities into urban, suburban, and rural for easier decision-making.
   - Integration with the Cost of Living API to provide updated data for budgeting.

### 5. **Interactive Map**
   - Google Maps integration for exploring points of interest (POIs) in Portugal.
   - Visualized data to help migrants familiarize themselves with their new environment.

### 6. **User-Centric Design**
   - User-friendly interface with routes for housing, directory, and chatbot services.
   - Responsive design ensuring accessibility across devices.

---

## Tech Stack
### Backend
- **Flask**: Lightweight and flexible framework powering the API and routing.
- **OpenAI**: Chatbot capabilities enhanced with GPT-3.5-turbo for natural and conversational responses.
- **Google Translate API**: Automatic translation for multilingual users.
- **Idealista API**: Real-time housing data.
- **Cost of Living API**: Provides comprehensive insights into city-specific expenses.

### Frontend
- HTML/CSS templates for:
  - Chatbot interaction
  - Housing preferences
  - Directory exploration
  - Points of interest visualization

### Data Sources
- **Static Data**: JSON files for office directories and embassy locations.
- **Dynamic Data**: APIs for cost of living and property search.

---

## Installation
### Prerequisites
1. Python 3.9 or higher.
2. Virtual environment tool (e.g., `venv`).
3. API Keys:
   - OpenAI API
   - Google Translate API
   - Idealista API
   - Cost of Living API

### Steps
1. Clone the repository:

   ```bash
   git clone https://github.com/guada-aguero/bureaucrat-butler.git
   cd bureaucrat-butler

2. Create and activate a virtual environment:

   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate

3. Install dependencies: 
   
   pip install -r requirements.txt

4. Set up environment variables:

   - Create a .env file in the root directory with the following:

     OPENAI_API_KEY=your_openai_key
     IDEALISTA_API_KEY=your_idealista_key
     IDEALISTA_API_SECRET=your_idealista_secret
     COST_OF_LIVING_API_KEY=your_cost_of_living_key
     GOOGLE_MAPS_API_KEY=your_google_maps_key

5. Run the application:

   python app.py

6. Access the app: Visit http://127.0.0.1:5000 in your browser.


## Usage

### Routes
- **Home Page**: `/`
- **Housing Recommendations**: `/housing`
- **Directory of Offices**: `/directory`
- **Interactive Map**: `/poi-map`
- **Dashboard for Cost of Living**: `/dashboard`
- **Chatbot**: `/ask`

### Workflow
1. Use the chatbot to ask questions about visas, NIF, and NISS.
2. Explore housing options by providing preferences such as budget and location.
3. Navigate through public office directories for guidance on government services.
4. Review cost-of-living data and urban insights to finalize relocation plans.

---

## Challenges and Vision

### Challenges
- Centralizing scattered information into one cohesive platform.
- Ensuring data reliability and up-to-date responses for legal and procedural queries.

### Vision
- Expand the AI Migration Assistant to support multiple countries, creating a global tool that simplifies migration for millions worldwide.
- Partner with more comprehensive APIs like **Numbeo** to enhance cost-of-living insights and improve data accuracy.
- Automate updates to the knowledge base (KB) to ensure real-time reliability and relevance for users.
- Introduce new features like:
  - A **Job Directory** to help migrants find work opportunities in their destination country.
  - A **Checklist Generator** for personalized relocation plans, including steps for visas, housing, and official registrations.
  - Integration with platforms like **LinkedIn** or local job boards to streamline job search.
  - Enhanced support for **language learning resources** to assist migrants in adapting linguistically to their new environment.

---

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request with a clear description of your changes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact
Developed by **Guadalupe Solis**  
GitHub: [guada-aguero](https://github.com/guada-aguero)
