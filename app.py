from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from utils.rag import load_knowledge_base, load_embedding_model, chapters_from_metadata, retrieve, infer_topic_chapters, retrieve_topics
from utils.question_generator import generate_paper, generate_single_question
from utils.validators import validate_paper
from utils.docx_exporter import build_docx, build_answer_key_docx
from utils.pdf_exporter import build_pdf
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='FBISE Physics AI Paper Generator',page_icon='📘',layout='wide')
st.title('FBISE Physics AI Paper Generator')
st.caption('Textbook-grounded • FBISE-calibrated • Simple Test or FLP • Compact printing')
for k in ['GROQ_API_KEY','LLM_API_KEY','LLM_MODEL','LLM_BASE_URL']:
    try:
        if k in st.secrets: os.environ[k]=str(st.secrets[k])
    except Exception: pass
@st.cache_resource
def emb(): return load_embedding_model()
@st.cache_resource
def kb(y): return load_knowledge_base(ROOT/'vector_store'/y)

def preview(p,s):
    st.subheader('Paper Preview')
    if s['paper_type']=='FLP Paper':
        st.markdown(f"### Physics {s['level']} - FLP"); st.markdown('#### SECTION-A (OBJECTIVE)'); st.write(f"Q No 1: MULTIPLE CHOICE QUESTIONS [{s['mcq_marks']}×{len(p['mcqs'])}={s['mcq_marks']*len(p['mcqs'])}]")
    else:
        st.markdown(f"### FSC {'PART-I' if s['level']=='HSSC-I' else 'PART-II'} (FEDERAL)"); st.write(('Topics: '+s.get('topics','')) if s.get('scope_mode')=='Topic-wise test' else ('Chapters ('+', '.join(s['chapters'])+')')); st.markdown('#### Encircle the right answer')
    for i,q in enumerate(p['mcqs'],1): st.markdown(f"**{i}. {q['question']}**  \nA. {q['options']['A']}    B. {q['options']['B']}    C. {q['options']['C']}    D. {q['options']['D']}")
    st.markdown('#### SECTION-B (SHORT QUESTIONS)' if s['paper_type']=='FLP Paper' else '#### Short questions')
    st.caption(f"Attempt {'any '+str(s['short_attempt']) if s['short_attempt']<len(p['short_questions']) else 'all'} questions. Each question carries {s['short_marks']} marks.")
    for i,q in enumerate(p['short_questions'],1):
        st.write(f"{i}. {q['question']}")
        if q.get('or_question'): st.markdown('**OR**'); st.write(q['or_question'])
    st.markdown('#### SECTION-C (LONG QUESTIONS)' if s['paper_type']=='FLP Paper' else '#### Long questions')
    for i,q in enumerate(p['long_questions'],1):
        st.write(f"{i}. {q['question']}")
        if q.get('or_question'): st.markdown('**OR**'); st.write(q['or_question'])

level_label=st.selectbox('1. Class / Level',['FSC Part-I / HSSC-I / First Year','FSC Part-II / HSSC-II / Second Year']); year='first_year' if 'Part-I' in level_label else 'second_year'; level='HSSC-I' if year=='first_year' else 'HSSC-II'
try: index,metadata=kb(year); available=chapters_from_metadata(metadata)
except Exception as e: st.error(str(e)); st.info('First run `python ingest.py`, commit vector_store to GitHub, then deploy.'); st.stop()
scope_mode=st.radio('2. What do you want the test from?',['Chapter-wise test','Topic-wise test'],horizontal=True,help='Chapter-wise: questions may come from anywhere in the selected chapter(s). Topic-wise: type topic names and the app finds their chapter(s) from the selected textbook automatically.')
if scope_mode=='Chapter-wise test':
    if not available:
        st.error('No chapter headings were detected in this textbook index. Run the updated `python ingest.py` once, then refresh this page.')
    chapters=st.multiselect('Select chapter(s)',available,default=available[:1])
    topic_text=''
    st.caption('Complete chapter means questions can be generated from any supported content anywhere in the selected chapter(s).')
else:
    chapters=[]
    topic_text=st.text_area('Type the topic name(s)',placeholder='Examples: projectile motion; angular momentum; Bernoulli equation',help='You do not need to know the chapter number. Enter topics that exist in the selected Physics textbook. The app searches the book and identifies the relevant chapter automatically.')
    st.caption('The app will search the selected class textbook, identify which chapter contains the topic, and use only the relevant textbook passages to build the paper.')
paper_type=st.radio('3. Paper pattern',['Simple Test','FLP Paper'],horizontal=True,help='Simple follows the Chapter 1,2,3,4,5 sample. FLP follows the supplied FLP structure.')
st.info('Selected output pattern: **'+paper_type+'**')
difficulty=st.radio('4. Difficulty',['FBISE Standard','Easy','Moderate','Challenging'],horizontal=True,index=0)
st.caption('FBISE Standard is calibrated to your reference papers: concise conceptual/reasoning questions, applications, dimensional checks, derivations and solvable numericals.')
c1,c2,c3=st.columns(3)
with c1: mcq_count=st.number_input('MCQs',0,50,10); mcq_marks=st.number_input('Marks / MCQ',1,10,1)
with c2: short_count=st.number_input('Short questions',0,30,11); short_marks=st.number_input('Marks / short',1,20,3)
with c3: long_count=st.number_input('Long questions',0,15,3); long_marks=st.number_input('Marks / long',1,30,5)
st.markdown('#### 5. Student choice / OR pattern')
or_mode=st.radio('OR choices',['No OR choices','OR in Short Questions only','OR in Long Questions only','OR in both Short and Long Questions'],horizontal=True)
or_short=or_mode in ['OR in Short Questions only','OR in both Short and Long Questions']; or_long=or_mode in ['OR in Long Questions only','OR in both Short and Long Questions']
if paper_type=='Simple Test' and or_mode=='No OR choices': st.caption('Result: compact Simple pattern like the Chapter 1,2,3,4,5 document, without alternatives.')
elif paper_type=='Simple Test': st.caption('Result: Simple-test layout with explicit OR alternatives, like the supplied FBISE reference style.')
elif or_mode=='No OR choices': st.caption('Result: FLP section structure, but no OR alternatives.')
else: st.caption('Result: FLP section structure with the selected OR alternatives.')
balanced=st.checkbox('Balanced chapter distribution',True)
ca,cb=st.columns(2)
with ca: short_attempt=st.number_input('Short questions students must attempt',0,int(short_count),int(short_count))
with cb: long_attempt=st.number_input('Long questions students must attempt',0,int(long_count),int(long_count))
scenario=st.text_area('6. Describe the test you want',placeholder='Focus on projectile motion and rotational dynamics; prefer conceptual questions.')
answer_key=st.checkbox('Generate Teacher Answer Key')
s={'level':level,'chapters':chapters,'scope_mode':scope_mode,'topics':topic_text,'paper_type':paper_type,'difficulty':difficulty,'mcq_count':int(mcq_count),'short_count':int(short_count),'long_count':int(long_count),'mcq_marks':int(mcq_marks),'short_marks':int(short_marks),'long_marks':int(long_marks),'or_short':or_short,'or_long':or_long,'balanced':balanced,'short_attempt':int(short_attempt),'long_attempt':int(long_attempt),'scenario':scenario}
ready = bool(chapters) if scope_mode=='Chapter-wise test' else bool(topic_text.strip())
if st.button('Generate Paper',type='primary',disabled=not ready):
    try:
        wanted_k = max(
    10,
    min(
        18,
        int(mcq_count) + int(short_count) + int(long_count)
    )
)
        if scope_mode=='Chapter-wise test':
            q=f"Physics assessment evidence from {', '.join(chapters)}. {scenario} definitions laws principles applications examples derivations numericals conceptual exercise"
            chunks=retrieve(q,index,metadata,emb(),chapters,top_k=wanted_k)
            effective_chapters=chapters
            if not chunks: raise RuntimeError('No textbook evidence found for those chapters.')
        else:
            # Search EACH topic independently across the entire selected textbook. Chapter metadata
            # is helpful for display, but is no longer required for topic-wise generation.
            chunks, topic_map = retrieve_topics(
    topic_text.strip(),
    index,
    metadata,
    emb(),
    per_topic=5,
    total_limit=min(14, wanted_k),
)
            if scenario.strip():
                extra=retrieve(scenario.strip(),index,metadata,emb(),chapters=None,top_k=3)
                known={(c.get('page'),c.get('chunk_id')) for c in chunks}
                chunks += [c for c in extra if (c.get('page'),c.get('chunk_id')) not in known]
            if not chunks:
                raise RuntimeError('No matching textbook passages were found for those topics.')
            inferred=sorted({c.get('chapter') for c in chunks if c.get('chapter') not in (None,'Unknown')})
            effective_chapters=inferred
            s['chapters']=effective_chapters
            if inferred:
                st.info('Relevant textbook material found in: **' + ', '.join(inferred) + '**. The paper will stay focused on: **' + topic_text.strip() + '**.')
            else:
                st.info('Relevant textbook passages were found. The PDF chapter label was not readable, but topic-wise generation can continue from the matched textbook pages.')
        with st.spinner('Retrieving textbook evidence and generating FBISE-style paper...'): p=generate_paper(chunks,s)
        errs = validate_paper(
    p,
    (int(mcq_count), int(short_count), int(long_count)),
    effective_chapters,
    scope_mode,
)
        st.session_state.update(paper=p,settings=s,chunks=chunks)
        if errs: st.warning('Review suggested: '+' | '.join(errs[:6]))
    except Exception as e: st.error(str(e))
if 'paper' in st.session_state:
    p=st.session_state.paper; s=st.session_state.settings; preview(p,s)
    with st.expander('Teacher grounding / textbook sources'):
        for c in st.session_state.chunks[:12]: st.caption(f"{c['chapter']} • p.{c['page']} • similarity {c['score']:.3f}"); st.write(c['text'][:550])
    a,b,c=st.columns(3); a.download_button('Download Word',build_docx(p,s),'FBISE_Physics_Paper.docx'); b.download_button('Download PDF',build_pdf(p,s),'FBISE_Physics_Paper.pdf','application/pdf')
    if answer_key: c.download_button('Download Answer Key',build_answer_key_docx(p,s),'Teacher_Answer_Key.docx')
st.divider(); st.subheader('Generate one question')
topic=st.text_input('Topic / scenario',placeholder='Projectile motion of a football'); mode=st.radio('Question style',['Simple Question','FLP-Style Question'],horizontal=True)
if st.button('Generate One Question',disabled=not topic):
    try:
        hits,_=retrieve_topics(topic,index,metadata,emb(),per_topic=10,total_limit=10)
        if not hits: raise RuntimeError('Topic not found in the selected textbook.')
        inferred=sorted({h.get('chapter') for h in hits if h.get('chapter') not in (None,'Unknown')})
        st.caption('Detected chapter: '+(', '.join(inferred) if inferred else 'chapter label unavailable; matched textbook pages used'))
        st.success(generate_single_question(hits,level,topic,mode)['question'])
    except Exception as e: st.error(str(e))
