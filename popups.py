import streamlit as st

def power_analysis_popup():
    if st.button("Why is the default sample size 30?"):
        st.session_state.show_power_help = True

    if st.session_state.get("show_power_help", False):
        with st.expander("Power analysis for masked vs original sentiment", expanded=True):

            

            st.markdown("""
By default, we assume a medium effect size is a practical compromise:

- Small effects (~0.2) need ~200 sentences → **lots of LLM sentiment calls → high cost**  
- Large effects (~0.8) need ~15 sentences → but such shifts are uncommon  
- Medium effect (~0.5) needs ~30 sentences → **good power without overspending LLM tokens**
""")

