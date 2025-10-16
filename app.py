import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (.env for local; secrets recommended for deployment)
load_dotenv()
# For Streamlit Cloud, use: api_key = st.secrets["GEMINI_API_KEY"]
api_key = os.getenv("GEMINI_API_KEY")

def load_css():
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.5rem; color: #2e86ab; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #2e86ab; padding-bottom: 0.5rem; }
    .info-box {
        background-color: #f0f2f6 !important;
        color: #212529 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border-left: 4px solid #2e86ab !important;
        margin-bottom: 1rem !important;
    }
    .success-box {
        background-color: #d4edda !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        color: #212529 !important;
    }
    .legal-section {
        background-color: #e8f4f8 !important;
        color: #212529 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .accused-card {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        color: #212529 !important;
    }
    </style>
    """, unsafe_allow_html=True)

class GeminiModelManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.selected_model_name = None
        self.configured = False
        self._ensure_configured()
    def _ensure_configured(self):
        if self.api_key and self.api_key != "YOUR_API_KEY_HERE":
            try:
                genai.configure(api_key=self.api_key)
                self.configured = True
            except Exception:
                self.configured = False
    def list_models(self) -> List[str]:
        if not self.configured:
            return []
        try:
            models = genai.list_models()
            return [m.name for m in models if getattr(m, "name", None)]
        except Exception:
            return []
    def pick_working_model(self) -> str:
        if not self.configured:
            raise RuntimeError("Gemini API not configured")
        models = self.list_models()
        candidates = [name for name in models if "gemini" in name.lower()]
        def score(n: str) -> float:
            ver = 0.0
            import re
            m = re.search(r'(\d+(?:\.\d+)?)', n)
            if m:
                try: ver = float(m.group(1))
                except Exception: ver = 0.0
            return ver
        candidates.sort(key=lambda x: score(x), reverse=True)
        for candidate in candidates:
            try:
                model = genai.GenerativeModel(candidate)
                test_response = model.generate_content("Hello")
                self.selected_model_name = candidate
                return candidate
            except Exception:
                continue
        raise RuntimeError("No working Gemini model found for generateContent")
    def get_selected_model(self) -> str:
        if self.selected_model_name:
            return self.selected_model_name
        try:
            self.selected_model_name = self.pick_working_model()
            return self.selected_model_name
        except Exception:
            return ""

class GeminiFIAnalyzer:
    def __init__(self, api_key: str):
        self.model_manager = GeminiModelManager(api_key)
    
    def analyze_with_gemini(self, fir_text: str) -> Dict[str, any]:
        if not self.model_manager.configured:
            return self._fallback_response("Gemini API not configured")
        model_name = self.model_manager.get_selected_model()
        if not model_name:
            try:
                model_name = self.model_manager.pick_working_model()
            except Exception as e:
                return self._fallback_response(f"Gemini model error: {str(e)}")
        try:
            model = genai.GenerativeModel(model_name)
            prompt = f"""
You are an AI legal expert specialized in Indian criminal law. Analyze the following FIR text and extract structured information. 
Map the offences to the CORRECT legal sections from Bharatiya Nyaya Sanhita (BNS) 2023, which replaced the IPC.

CRITICAL INSTRUCTION: Use ONLY BNS 2023 sections with their CORRECT definitions. DO NOT confuse IPC sections with BNS sections.

KEY BNS 2023 SECTIONS (Use these as reference):
- Section 103: Murder (not 309)
- Section 115(2): Voluntarily causing hurt
- Section 117: Voluntarily causing grievous hurt
- Section 309(2): Robbery (taking property with threat/force)
- Section 310: Dacoity
- Section 351(2): Criminal intimidation (threats to cause harm)
- Section 351(3): Criminal intimidation with threat to cause death
- Section 352: Intentional insult with intent to provoke breach of peace
- Section 356(2): Theft
- Section 303(2): Kidnapping

SC/ST (PREVENTION OF ATROCITIES) ACT, 1989:
- Section 3(1)(r): Intentionally insults or intimidates with intent to humiliate a member of SC/ST in any place within public view
- Section 3(1)(s): Abuses any member of SC/ST by caste name in any place within public view
- Section 3(2)(v): Commits any offence under the Act on the ground that such person is a member of SC/ST

ARMS ACT, 1959:
- Section 25: Possession/use of prohibited arms and ammunition
- Section 27: Use of arms in commission of an offence

MOTOR VEHICLES ACT, 1988:
- Section 66: Use of vehicle without registration or in violation of rules

FIR TEXT:
{fir_text}

EXTRACTION REQUIREMENTS:
1. Extract all relevant details: Complainant, Accused, Date/Time, Place, Vehicles, Weapons, Offences, Injuries, etc.
2. For legal mapping, identify the criminal acts committed and map them to:
   - BNS 2023 sections (with correct section numbers and descriptions)
   - SC/ST Atrocities Act (if caste-based abuse/discrimination is present)
   - Arms Act (if firearms/weapons are used)
   - Motor Vehicles Act (if vehicles were used unlawfully)
3. Provide detailed legal analysis explaining why each section applies.

IMPORTANT: 
- BNS Section 309 is for ROBBERY (not murder)
- BNS Section 103 is for MURDER
- Double-check section numbers against the definitions provided above
- If you're unsure about a section, provide your best legal reasoning

OUTPUT FORMAT (JSON ONLY, NO MARKDOWN):
{{
    "extracted_info": {{
        "Complainant": {{
            "Name": "string",
            "Father": "string",
            "Age": number,
            "Community": "string",
            "Occupation": "string",
            "Address": "string"
        }},
        "DateTime": "string",
        "Place": "string",
        "Accused": [{{
            "Name": "string",
            "Age": number,
            "Relation": "string",
            "Occupation": "string",
            "Address": "string",
            "History": "string"
        }}],
        "Vehicles": ["string"],
        "WeaponsUsed": ["string"],
        "Offences": ["string"],
        "Injuries": "string",
        "PropertyLoss": ["string"],
        "Threats": ["string"],
        "Witnesses": ["string"],
        "Impact": "string"
    }},
    "legal_mapping": {{
        "BNS 2023": [
            "Section XXX - Description: Explanation of why this section applies"
        ],
        "SC/ST Atrocities Act, 1989": [
            "Section X(X)(x) - Description: Explanation"
        ],
        "Arms Act, 1959": [
            "Section XX - Description: Explanation"
        ],
        "Motor Vehicles Act, 1988": [
            "Section XX - Description: Explanation"
        ]
    }},
    "legal_analysis": "Comprehensive analysis explaining: 1) What criminal acts occurred, 2) Which specific BNS 2023 sections apply and why (with correct section numbers), 3) Applicability of special acts (SC/ST, Arms, MVA), 4) Severity assessment, 5) Recommended charges"
}}

Respond with ONLY the JSON object, no markdown formatting or code blocks.
"""
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response text
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Extract JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_text = response_text[start_idx:end_idx]
            else:
                json_text = response_text
            
            try:
                result = json.loads(json_text)
                result['timestamp'] = datetime.now().isoformat()
                result['model_used'] = model_name
                return result
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                return self.extract_from_text_response(response_text, fir_text, model_name)
        except Exception as e:
            return self._fallback_response(f"Gemini API error: {str(e)}")
    
    def extract_from_text_response(self, response_text: str, original_fir: str, model_name: str) -> Dict[str, any]:
        extracted_info = {
            "Complainant": {},
            "DateTime": "",
            "Place": "",
            "Accused": [],
            "Vehicles": [],
            "WeaponsUsed": [],
            "Offences": [],
            "Injuries": "",
            "PropertyLoss": [],
            "Threats": [],
            "Witnesses": [],
            "Impact": ""
        }
        legal_mapping = {
            "BNS 2023": [],
            "SC/ST Atrocities Act, 1989": [],
            "Arms Act, 1959": [],
            "Motor Vehicles Act, 1988": []
        }
        legal_analysis = f"Analysis completed with fallback method using model {model_name}. Please review the extracted information manually."
        
        import re
        # Extract vehicle numbers
        vehicles = re.findall(r'[A-Z]{2}-\d{2}-[A-Z]{1,2}-\d{4}', original_fir)
        extracted_info["Vehicles"] = vehicles
        
        # Extract weapons
        weapons_keywords = ['knife', 'pistol', 'gun', 'rod', 'stick', 'weapon', 'firearm']
        for weapon in weapons_keywords:
            if weapon in original_fir.lower():
                extracted_info["WeaponsUsed"].append(weapon)
        
        # Basic legal mapping based on keywords
        if any(word in original_fir.lower() for word in ['robbery', 'snatched', 'forcibly took']):
            legal_mapping["BNS 2023"].append("Section 309(2) - Robbery: Forcible taking of property")
        
        if any(word in original_fir.lower() for word in ['threatened', 'threat', 'intimidat']):
            legal_mapping["BNS 2023"].append("Section 351(2) - Criminal intimidation")
        
        if any(word in original_fir.lower() for word in ['injury', 'hurt', 'injured', 'bleeding']):
            legal_mapping["BNS 2023"].append("Section 115(2) - Voluntarily causing hurt")
        
        if any(word in original_fir.lower() for word in ['caste', 'scheduled caste', 'sc/st']):
            legal_mapping["SC/ST Atrocities Act, 1989"].append("Section 3(1)(r) - Intentional insult/abuse")
        
        if 'pistol' in original_fir.lower() or 'gun' in original_fir.lower() or 'firearm' in original_fir.lower():
            legal_mapping["Arms Act, 1959"].append("Section 25 - Possession/use of prohibited arms")
            legal_mapping["Arms Act, 1959"].append("Section 27 - Use of arms in commission of offence")
        
        return {
            "extracted_info": extracted_info,
            "legal_mapping": legal_mapping,
            "legal_analysis": legal_analysis,
            "timestamp": datetime.now().isoformat(),
            "model_used": model_name
        }
    
    def _fallback_response(self, error_msg: str) -> Dict[str, any]:
        return {
            "extracted_info": {
                "Complainant": {"Name": "Analysis Failed", "Error": error_msg},
                "DateTime": "N/A",
                "Place": "N/A",
                "Accused": [],
                "Vehicles": [],
                "WeaponsUsed": [],
                "Offences": [],
                "Injuries": "N/A",
                "PropertyLoss": [],
                "Threats": [],
                "Witnesses": [],
                "Impact": f"Analysis failed - {error_msg}"
            },
            "legal_mapping": {
                "BNS 2023": [],
                "SC/ST Atrocities Act, 1989": [],
                "Arms Act, 1959": [],
                "Motor Vehicles Act, 1988": []
            },
            "legal_analysis": f"Analysis could not be completed. Error: {error_msg}",
            "timestamp": datetime.now().isoformat(),
            "model_used": "none"
        }

class DharmaFIRAnalyzer:
    def __init__(self, api_key: str):
        self.gemini = GeminiFIAnalyzer(api_key)
    def analyze_fir(self, fir_text: str) -> Dict[str, any]:
        return self.gemini.analyze_with_gemini(fir_text)
    def display_results(self, results: Dict[str, any]):
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success("✅ FIR Analysis Complete using Gemini AI!")
        if results.get('model_used'):
            st.info(f"Model used: {results['model_used']}")
        st.markdown('</div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Extracted Information", "⚖️ Legal Mapping", "🔍 Detailed Analysis", "📊 Summary"])
        with tab1:
            self._display_extracted_info(results.get('extracted_info', {}))
        with tab2:
            self._display_legal_mapping(results.get('legal_mapping', {}))
        with tab3:
            self._display_detailed_analysis(results)
        with tab4:
            self._display_summary(results)
    def _display_extracted_info(self, extracted_info: Dict[str, Any]):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">👤 Complainant Information</div>', unsafe_allow_html=True)
            complainant = extracted_info.get('Complainant', {})
            if complainant and 'Error' not in complainant:
                for key, value in complainant.items():
                    if value:
                        st.write(f"**{key}:** {value}")
            else:
                st.write("No complainant information extracted")
            st.markdown('<div class="section-header">📅 Incident Details</div>', unsafe_allow_html=True)
            st.write(f"**Date & Time:** {extracted_info.get('DateTime', 'N/A')}")
            st.write(f"**Place:** {extracted_info.get('Place', 'N/A')}")
            st.write(f"**Injuries:** {extracted_info.get('Injuries', 'N/A')}")
            st.write(f"**Impact:** {extracted_info.get('Impact', 'N/A')}")
        with col2:
            st.markdown('<div class="section-header">🚨 Accused Persons</div>', unsafe_allow_html=True)
            accused_list = extracted_info.get('Accused', [])
            if accused_list:
                for accused in accused_list:
                    st.markdown(f'<div class="accused-card">', unsafe_allow_html=True)
                    st.write(f"**Name:** {accused.get('Name', 'N/A')}")
                    for key, value in accused.items():
                        if key != 'Name' and value:
                            st.write(f"  **{key}:** {value}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.write("No accused information extracted")
    def _display_legal_mapping(self, legal_mapping: Dict[str, List[str]]):
        st.markdown('<div class="section-header">⚖️ Legal Sections Applied</div>', unsafe_allow_html=True)
        if not legal_mapping or all(not sections for sections in legal_mapping.values()):
            st.warning("No legal sections could be mapped.")
            return
        for act, sections in legal_mapping.items():
            if sections:
                st.subheader(f"{act}")
                for section in sections:
                    st.markdown(f'<div class="legal-section">{section}</div>', unsafe_allow_html=True)
    def _display_detailed_analysis(self, results: Dict[str, any]):
        extracted_info = results.get('extracted_info', {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">⚖️ Offences & Evidence</div>', unsafe_allow_html=True)
            st.write("**Offences Identified:**")
            offences = extracted_info.get('Offences', [])
            if offences:
                for offence in offences:
                    st.write(f"• {offence}")
            else:
                st.write("No specific offences identified")
            st.write("**🚗 Vehicles:**")
            vehicles = extracted_info.get('Vehicles', [])
            if vehicles:
                for v in vehicles:
                    st.write(f"- {v}")
            else:
                st.write("- No vehicles identified")
            st.write("**🔫 Weapons Used:**")
            weapons = extracted_info.get('WeaponsUsed', [])
            if weapons:
                for w in weapons:
                    st.write(f"- {w}")
            else:
                st.write("- No weapons identified")
        with col2:
            st.markdown('<div class="section-header">👥 Witnesses & Impact</div>', unsafe_allow_html=True)
            st.write("**Witnesses:**")
            witnesses = extracted_info.get('Witnesses', [])
            if witnesses:
                for w in witnesses:
                    st.write(f"• {w}")
            else:
                st.write("No witnesses identified")
            st.write("**💰 Property Loss:**")
            property_loss = extracted_info.get('PropertyLoss', [])
            if property_loss:
                for p in property_loss:
                    st.write(f"- {p}")
            else:
                st.write("- No property loss identified")
            st.write("**⚠️ Threats:**")
            threats = extracted_info.get('Threats', [])
            if threats:
                for t in threats:
                    st.write(f"• {t}")
            else:
                st.write("No specific threats identified")
    def _display_summary(self, results: Dict[str, any]):
        extracted_info = results.get('extracted_info', {})
        legal_mapping = results.get('legal_mapping', {})
        total_accused = len(extracted_info.get('Accused', []))
        total_offences = len(extracted_info.get('Offences', []))
        total_legal_sections = sum(len(sections) for sections in legal_mapping.values())
        total_acts = sum(1 for sections in legal_mapping.values() if sections)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👤 Accused Persons", total_accused)
        with col2:
            st.metric("⚖️ Offences Identified", total_offences)
        with col3:
            st.metric("📚 Legal Sections", total_legal_sections)
        with col4:
            st.metric("📜 Acts Violated", total_acts)
        st.markdown("### 📝 Legal Analysis")
        st.markdown(results.get('legal_analysis', 'No analysis available'))
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download Analysis Results (JSON)",
            data=json_str,
            file_name=f"fir_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    load_css()
    st.markdown('<div class="main-header">⚖️ DHARMA FIR Analyzer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>AI-powered analysis using Google Gemini AI</strong><br>
    Extract structured information and map to relevant legal sections (BNS 2023) from mixed English-Telugu FIR texts.
    </div>
    """, unsafe_allow_html=True)

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        st.error("Gemini API Key not found. Please set GEMINI_API_KEY in your .env file or Streamlit secrets.")
        st.stop()

    analyzer = DharmaFIRAnalyzer(api_key)

    sample_fir = """On 5th September 2025, at about 11:45 PM, I, Praveen Kumar, S/o Srinivas Rao, aged 35 years, belonging to BC community, working as a private school teacher, residing at Gandhi Nagar, Hyderabad, was returning home on my Hero Splendor motorcycle (TS-09-FQ-5678). Near the RTC Crossroads, two persons stopped me. One, later identified as Raju Singh, age around 32, resident of Malkajgiri, threatened me with a knife and demanded my wallet and phone. The other, Mohan, S/o Shankar, age 28, hit me on the arm with an iron rod and took my smartwatch worth ₹12,000.

When I tried to shout for help, Raju threatened, "Reporting to police will get you and your family in trouble." Local tea stall owner Ramulu and auto driver Venkatesh witnessed the incident but were afraid to intervene.

During the scuffle, my arm was injured. The accused escaped on a white Activa (TS-10-AB-3210). I suffered mental trauma and fear for my family's safety.

ఈ దాడి వల్ల నాకు గాయాలు అయ్యాయి మరియు నాకు మానసిక భయంతో పాటు ఆస్తి నష్టం కూడా జరిగింది. నేను వెంటనే పోలీసుగా ఫిర్యాదు చేస్తున్నాను."""
    st.markdown('<div class="section-header">📝 FIR Text Input</div>', unsafe_allow_html=True)
    input_method = st.radio(
        "Choose input method:",
        ["Use Sample FIR", "Paste Your Own FIR Text"], horizontal=True
    )
    fir_text = ""
    if input_method == "Use Sample FIR":
        fir_text = sample_fir
        st.text_area("FIR Text", fir_text, height=300, key="sample_fir", label_visibility="collapsed")
    else:
        fir_text = st.text_area(
            "Paste your FIR text (English + Telugu mixed):",
            height=300,
            placeholder="Paste the police complaint text here...",
            key="custom_fir",
            label_visibility="collapsed"
        )
    if st.button("🔍 Analyze FIR with Gemini AI", type="primary", use_container_width=True):
        if fir_text.strip():
            with st.spinner("Analyzing FIR text with Gemini AI... This may take 10-20 seconds."):
                try:
                    results = analyzer.analyze_fir(fir_text)
                    analyzer.display_results(results)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please enter FIR text to analyze.")
    with st.sidebar:
        st.markdown("### 🏛️ About DHARMA Project")
        st.info("""
        This AI system uses Google Gemini AI to:
        - Extract structured information from FIRs
        - Map offences to legal sections (BNS 2023)
        - Handle multilingual text (English + Telugu)
        - Generate comprehensive legal analysis
        """)
        st.markdown("### 📊 Key BNS 2023 Sections")
        st.write("""
        - **Sec 103**: Murder
        - **Sec 115(2)**: Voluntarily causing hurt
        - **Sec 309(2)**: Robbery
        - **Sec 351**: Criminal intimidation
        - **Sec 352**: Intentional insult
        """)
        st.markdown("### 📚 Other Acts")
        st.write("""
        - SC/ST Atrocities Act, 1989
        - Arms Act, 1959
        - Motor Vehicles Act, 1988
        """)

if __name__ == "__main__":
    main()
