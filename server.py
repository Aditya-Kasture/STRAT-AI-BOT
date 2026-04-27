#!/usr/bin/env python3
"""
Strat AI Audit & Scoping Bot — Complete Backend Server
Revision: April 2026 — All 26 items from revision checklist implemented.
Run: pip install -r requirements.txt && python server.py
"""

import os
import json
import uuid
import asyncio
import logging
import re
import time
import base64
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx

# ChromaDB for RAG (uses built-in sentence-transformer embeddings — no OpenAI key needed)
try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("strat-ai")

# ===================================================================
# CONFIG
# ===================================================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./data")
FRONTEND_DIR = os.getenv("FRONTEND_DIR", "./frontend")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
MAX_HISTORY = 30
SESSION_TTL_HOURS = 72
BATCH_SIZE = 6

# Item 18: HubSpot
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY", "")
# Item 23: Slack
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
# Item 19: Calendly — set both URLs in .env (Yaseen to provide)
CALENDLY_URL = os.getenv("CALENDLY_URL", "https://calendly.com/yaseen-stratai/30min")
CALENDLY_URL_INTRO = os.getenv("CALENDLY_URL_INTRO", CALENDLY_URL)  # alternate/intro link
# Public contact / website URL used in live summary panel footer (no direct phone)
STRAT_AI_WEBSITE_URL = os.getenv("STRAT_AI_WEBSITE_URL", "https://stratai.solutions/contact")
STRAT_AI_CONTACT_EMAIL = os.getenv("STRAT_AI_CONTACT_EMAIL", "yaseen@stratai.solutions")
# Item 20: Admin
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "stratai2026")
# Item 21: Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Initialise Supabase client once at startup (only if credentials are present)
_supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client as _sb_create
        _supabase_client = _sb_create(SUPABASE_URL, SUPABASE_KEY)
        logging.getLogger("strat-ai").info("Supabase client initialised")
    except Exception as _e:
        logging.getLogger("strat-ai").warning(f"Supabase init failed: {_e}")

# ===================================================================
# ENUMS & DATA MODELS
# ===================================================================
class Stage(str, Enum):
    INTAKE = "intake"           # Item 17: email capture gate
    CLASSIFY = "classify"
    QUALIFY = "qualify"
    SNAPSHOT = "snapshot"
    SYNTHESIS = "synthesis"
    DEEP_AUDIT = "deep_audit"
    PROPOSAL = "proposal_ready"
    COMPLETE = "complete"


class ClientType(str, Enum):
    BROKER = "broker"
    LENDER = "lender"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class Fit(str, Enum):
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    UNKNOWN = "unknown"


STAGE_ORDER = [
    Stage.INTAKE, Stage.CLASSIFY, Stage.QUALIFY, Stage.SNAPSHOT,
    Stage.SYNTHESIS, Stage.DEEP_AUDIT, Stage.PROPOSAL, Stage.COMPLETE,
]

# Item 10: Progress percentages tied to stage transitions
STAGE_PROGRESS = {
    Stage.INTAKE: 0,
    Stage.CLASSIFY: 16,
    Stage.QUALIFY: 33,
    Stage.SNAPSHOT: 50,
    Stage.SYNTHESIS: 66,
    Stage.DEEP_AUDIT: 83,
    Stage.PROPOSAL: 100,
    Stage.COMPLETE: 100,
}

STAGE_LABELS = {
    Stage.INTAKE: "Getting Started",
    Stage.CLASSIFY: "Classification",
    Stage.QUALIFY: "Qualification",
    Stage.SNAPSHOT: "Snapshot Audit",
    Stage.SYNTHESIS: "Synthesis",
    Stage.DEEP_AUDIT: "Deep Audit",
    Stage.PROPOSAL: "Proposal Ready",
    Stage.COMPLETE: "Complete",
}


@dataclass
class Session:
    id: str = ""
    created_at: str = ""
    stage: Stage = Stage.INTAKE      # Start at intake for email capture
    client_type: ClientType = ClientType.UNKNOWN
    fit: Fit = Fit.UNKNOWN
    # Item 17: Contact info
    contact_name: str = ""
    contact_email: str = ""
    company_name: str = ""
    # Audit data
    qual_data: Dict[str, Any] = field(default_factory=dict)
    snapshot_answers: Dict[str, Any] = field(default_factory=dict)
    snapshot_batch: int = 0
    synthesis_text: str = ""
    deep_modules: List[str] = field(default_factory=list)
    deep_module_idx: int = 0
    deep_answers: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Item 26: Cost tracking
    api_cost_usd: float = 0.0
    api_calls: int = 0
    # Item 19: Calendly tracking
    calendly_clicked: bool = False
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value if isinstance(self.stage, Enum) else self.stage
        d["client_type"] = (
            self.client_type.value if isinstance(self.client_type, Enum) else self.client_type
        )
        d["fit"] = self.fit.value if isinstance(self.fit, Enum) else self.fit
        return d

    def progress_pct(self) -> int:
        return STAGE_PROGRESS.get(self.stage, 0)


# ===================================================================
# QUESTION BANKS
# ===================================================================

BROKER_SNAPSHOT_QS = [
    {"id":"SB01","text":"In one sentence, what does your brokerage actually do today?","sec":"biz_overview","tags":["classification","identity"],"p":1},
    {"id":"SB02","text":"What type of deal do you want the system optimized for first?","sec":"biz_overview","tags":["scope","priority"],"p":1},
    {"id":"SB03","text":"What type of deal do you want the system to reject automatically?","sec":"biz_overview","tags":["qualification","auto_decline"],"p":2},
    {"id":"SB04","text":"If the system only fixed ONE thing in the next 30 days, what should it be?","sec":"biz_overview","tags":["priority","bottleneck","quick_win"],"p":1},
    {"id":"SB05","text":"What does a 'perfect deal' look like -- size, borrower, speed, complexity?","sec":"biz_overview","tags":["ideal_state","qualification"],"p":1},
    {"id":"SB06","text":"How many deals hit your desk in a typical week?","sec":"volume","tags":["volume","capacity"],"p":1},
    {"id":"SB07","text":"How many make it past your initial gut check?","sec":"volume","tags":["conversion","qualification"],"p":1},
    {"id":"SB08","text":"How many can you realistically handle without stress today?","sec":"volume","tags":["capacity","bottleneck"],"p":1},
    {"id":"SB09","text":"Where do deals bottleneck most often -- intake, docs, lenders, follow-up?","sec":"volume","tags":["bottleneck","pipeline"],"p":1},
    {"id":"SB10","text":"If volume doubled tomorrow, what would break first?","sec":"volume","tags":["scale","bottleneck","risk"],"p":1},
    {"id":"SB11","text":"List the stages a deal goes through from first contact to close.","sec":"stages","tags":["workflow","pipeline","stages"],"p":1},
    {"id":"SB12","text":"Where do humans have to be involved today?","sec":"stages","tags":["manual_steps","judgment"],"p":1},
    {"id":"SB13","text":"Where are humans involved only because 'that's how it's always been'?","sec":"stages","tags":["automation","legacy_process"],"p":1},
    {"id":"SB14","text":"What step do you personally touch on almost every deal?","sec":"stages","tags":["founder_bottleneck","delegation"],"p":1},
    {"id":"SB15","text":"What step do you wish you never had to touch again?","sec":"stages","tags":["delegation","automation"],"p":1},
    {"id":"SB16","text":"Where do you lose the most time per deal?","sec":"stages","tags":["time_loss","bottleneck"],"p":1},
    {"id":"SB17","text":"What is the single system you want to be the source of truth?","sec":"systems","tags":["source_of_truth"],"p":1},
    {"id":"SB18","text":"What systems are used today but shouldn't be?","sec":"systems","tags":["tech_debt","systems"],"p":2},
    {"id":"SB19","text":"Where does data get duplicated or re-entered?","sec":"systems","tags":["duplicate_entry","data_quality"],"p":1},
    {"id":"SB20","text":"Where does information live only in your head?","sec":"systems","tags":["knowledge_in_heads","risk"],"p":1},
    {"id":"SB21","text":"What tool do you open first every morning?","sec":"systems","tags":["source_of_truth","habits"],"p":2},
    {"id":"SB22","text":"What tool do you hate opening?","sec":"systems","tags":["tech_debt","adoption"],"p":2},
    {"id":"SB23","text":"How do deals most commonly come in today -- email, call, referral, form?","sec":"intake","tags":["intake","channels"],"p":1},
    {"id":"SB24","text":"What information do you need to make a YES/NO decision fast?","sec":"intake","tags":["qualification","decisioning"],"p":1},
    {"id":"SB25","text":"What's an automatic NO?","sec":"intake","tags":["qualification","auto_decline"],"p":1},
    {"id":"SB26","text":"What's an obvious YES?","sec":"intake","tags":["qualification","ideal_state"],"p":2},
    {"id":"SB27","text":"How long should a YES/NO decision take in the ideal world?","sec":"intake","tags":["cycle_time","qualification"],"p":2},
    {"id":"SB28","text":"What causes slow decisions today?","sec":"intake","tags":["bottleneck","cycle_time"],"p":1},
    {"id":"SB29","text":"What doc is always missing first?","sec":"docs","tags":["document_chasing","missing_docs"],"p":1},
    {"id":"SB30","text":"What doc causes the most follow-ups?","sec":"docs","tags":["document_chasing","follow_up"],"p":1},
    {"id":"SB31","text":"What does 'docs complete' mean to you?","sec":"docs","tags":["doc_collection","completeness"],"p":2},
    {"id":"SB32","text":"What part of packaging feels the most repetitive?","sec":"docs","tags":["automation","packaging"],"p":1},
    {"id":"SB33","text":"What part of packaging feels the most risky?","sec":"docs","tags":["risk","compliance","packaging"],"p":2},
    {"id":"SB34","text":"What would you trust automation to prepare automatically?","sec":"docs","tags":["automation","trust"],"p":2},
    {"id":"SB35","text":"How many lenders do you regularly work with?","sec":"lenders","tags":["lender_network"],"p":1},
    {"id":"SB36","text":"What % of deals go to direct lenders vs brokered?","sec":"lenders","tags":["lender_network","deal_flow"],"p":2},
    {"id":"SB37","text":"What info must be perfect before submission?","sec":"lenders","tags":["submission","quality"],"p":1},
    {"id":"SB38","text":"What follow-ups with lenders are pure busywork?","sec":"lenders","tags":["busywork","automation","follow_up"],"p":1},
    {"id":"SB39","text":"What lender communication must stay personal?","sec":"lenders","tags":["relationship","judgment"],"p":2},
    {"id":"SB40","text":"Who will use the system daily besides you?","sec":"team","tags":["users","adoption"],"p":1},
    {"id":"SB41","text":"Who will resist it most?","sec":"team","tags":["adoption","risk"],"p":2},
    {"id":"SB42","text":"Who can enforce usage if you're not around?","sec":"team","tags":["adoption","accountability"],"p":2},
    {"id":"SB43","text":"What bad habits do you want the system to kill?","sec":"team","tags":["adoption","process"],"p":2},
    {"id":"SB44","text":"What cannot be allowed to happen outside the system?","sec":"team","tags":["compliance","control"],"p":1},
    {"id":"SB45","text":"What would make staff love the system?","sec":"team","tags":["adoption","ux"],"p":2},
    {"id":"SB46","text":"How many brokers could you realistically support today?","sec":"scale","tags":["capacity","scale"],"p":2},
    {"id":"SB47","text":"In 6 months, how many do you want on the system?","sec":"scale","tags":["scale","growth"],"p":2},
    {"id":"SB48","text":"What would make you say 'we're not ready to add brokers yet'?","sec":"scale","tags":["readiness","risk"],"p":2},
    {"id":"SB49","text":"What would make you confident to onboard brokers aggressively?","sec":"scale","tags":["readiness","scale"],"p":2},
    {"id":"SB50","text":"What would be catastrophic if other brokers messed it up?","sec":"scale","tags":["risk","compliance"],"p":1},
    {"id":"SB51","text":"What assumptions do you think I might be making that could be wrong?","sec":"vision","tags":["trust","alignment"],"p":2},
    {"id":"SB52","text":"What scares you most about automating this?","sec":"vision","tags":["risk","trust"],"p":2},
    {"id":"SB53","text":"What excites you most about automating this?","sec":"vision","tags":["motivation"],"p":2},
    {"id":"SB54","text":"What would break trust in this partnership?","sec":"vision","tags":["trust","partnership"],"p":2},
    {"id":"SB55","text":"If this works perfectly, how does your day look different?","sec":"vision","tags":["ideal_state","motivation"],"p":1},
]

LENDER_SNAPSHOT_QS = [
    {"id":"SL01","text":"In one sentence, what does your lending operation actually do today?","sec":"biz_overview","tags":["classification","identity"],"p":1},
    {"id":"SL02","text":"What loan types do you want the system optimized for first?","sec":"biz_overview","tags":["scope","priority"],"p":1},
    {"id":"SL03","text":"What loan types do you want the system to screen out automatically?","sec":"biz_overview","tags":["qualification","auto_decline"],"p":2},
    {"id":"SL04","text":"If the system only fixed ONE thing in the next 30 days, what should it be?","sec":"biz_overview","tags":["priority","bottleneck","quick_win"],"p":1},
    {"id":"SL05","text":"What does a 'perfect loan' look like -- size, borrower type, property type, LTV, DSCR?","sec":"biz_overview","tags":["ideal_state","credit_box"],"p":1},
    {"id":"SL06","text":"How many loan applications hit your desk in a typical week?","sec":"volume","tags":["volume","capacity"],"p":1},
    {"id":"SL07","text":"How many make it past your initial credit screening?","sec":"volume","tags":["conversion","credit"],"p":1},
    {"id":"SL08","text":"How many loans can you realistically process without stress today?","sec":"volume","tags":["capacity","bottleneck"],"p":1},
    {"id":"SL09","text":"Where do loans bottleneck most often -- intake, underwriting, docs, closing?","sec":"volume","tags":["bottleneck","pipeline"],"p":1},
    {"id":"SL10","text":"If volume doubled tomorrow, what would break first?","sec":"volume","tags":["scale","bottleneck","risk"],"p":1},
    {"id":"SL11","text":"List the stages a loan goes through from application to funding.","sec":"stages","tags":["workflow","pipeline","stages"],"p":1},
    {"id":"SL12","text":"Where do humans have to be involved today?","sec":"stages","tags":["manual_steps","judgment"],"p":1},
    {"id":"SL13","text":"Where are humans involved only because 'that's how it's always been'?","sec":"stages","tags":["automation","legacy_process"],"p":1},
    {"id":"SL14","text":"What step do you personally touch on almost every loan?","sec":"stages","tags":["founder_bottleneck"],"p":1},
    {"id":"SL15","text":"What step do you wish you never had to touch again?","sec":"stages","tags":["delegation","automation"],"p":1},
    {"id":"SL16","text":"Where do you lose the most time per loan?","sec":"stages","tags":["time_loss","bottleneck"],"p":1},
    {"id":"SL17","text":"What is the single system you want to be the source of truth?","sec":"systems","tags":["source_of_truth"],"p":1},
    {"id":"SL18","text":"What systems are used today but shouldn't be?","sec":"systems","tags":["tech_debt"],"p":2},
    {"id":"SL19","text":"Where does data get duplicated or re-entered?","sec":"systems","tags":["duplicate_entry"],"p":1},
    {"id":"SL20","text":"Where does information live only in your head?","sec":"systems","tags":["knowledge_in_heads"],"p":1},
    {"id":"SL21","text":"What tool do you open first every morning?","sec":"systems","tags":["source_of_truth","habits"],"p":2},
    {"id":"SL22","text":"What tool do you hate opening?","sec":"systems","tags":["tech_debt","adoption"],"p":2},
    {"id":"SL23","text":"How do loan applications most commonly come in -- broker portal, direct borrower, email?","sec":"intake","tags":["intake","channels"],"p":1},
    {"id":"SL24","text":"What information do you need to make a YES/NO credit decision fast?","sec":"intake","tags":["qualification","credit","decisioning"],"p":1},
    {"id":"SL25","text":"What's an automatic decline?","sec":"intake","tags":["qualification","auto_decline","credit_box"],"p":1},
    {"id":"SL26","text":"What's an obvious approval?","sec":"intake","tags":["qualification","ideal_state"],"p":2},
    {"id":"SL27","text":"How long should a credit decision take in the ideal world?","sec":"intake","tags":["cycle_time"],"p":2},
    {"id":"SL28","text":"What causes slow credit decisions today?","sec":"intake","tags":["bottleneck","cycle_time"],"p":1},
    {"id":"SL29","text":"What document is always missing or incomplete first?","sec":"docs","tags":["document_chasing","missing_docs"],"p":1},
    {"id":"SL30","text":"What document causes the most follow-ups?","sec":"docs","tags":["document_chasing","follow_up"],"p":1},
    {"id":"SL31","text":"What does 'loan file complete' mean to you?","sec":"docs","tags":["doc_collection","completeness"],"p":2},
    {"id":"SL32","text":"What part of document review feels the most repetitive?","sec":"docs","tags":["automation"],"p":1},
    {"id":"SL33","text":"What part of document review feels the most risky -- compliance, fraud, errors?","sec":"docs","tags":["compliance","risk","fraud"],"p":1},
    {"id":"SL34","text":"What would you trust automation to verify or flag automatically?","sec":"docs","tags":["automation","trust"],"p":2},
    {"id":"SL35","text":"How many underwriters do you have?","sec":"underwriting","tags":["team","underwriting"],"p":1},
    {"id":"SL36","text":"What % of loans need senior underwriter review vs junior?","sec":"underwriting","tags":["underwriting","bottleneck"],"p":2},
    {"id":"SL37","text":"What info must be perfect before underwriting can start?","sec":"underwriting","tags":["quality","completeness"],"p":1},
    {"id":"SL38","text":"What back-and-forth with brokers/borrowers is pure busywork?","sec":"underwriting","tags":["busywork","automation"],"p":1},
    {"id":"SL39","text":"What underwriter communication must stay personal and judgment-based?","sec":"underwriting","tags":["judgment","relationship"],"p":2},
    {"id":"SL40","text":"Who will use the system daily besides you?","sec":"team","tags":["users","adoption"],"p":1},
    {"id":"SL41","text":"Who will resist it most?","sec":"team","tags":["adoption","risk"],"p":2},
    {"id":"SL42","text":"Who can enforce usage if you're not around?","sec":"team","tags":["adoption","accountability"],"p":2},
    {"id":"SL43","text":"What bad habits do you want the system to kill?","sec":"team","tags":["adoption","process"],"p":2},
    {"id":"SL44","text":"What cannot be allowed to happen outside the system?","sec":"team","tags":["compliance","control"],"p":1},
    {"id":"SL45","text":"What would make staff love the system?","sec":"team","tags":["adoption","ux"],"p":2},
    {"id":"SL46","text":"How many loans per month can you handle today?","sec":"scale","tags":["capacity"],"p":1},
    {"id":"SL47","text":"In 6 months, how many loans per month do you want to be funding?","sec":"scale","tags":["scale","growth"],"p":2},
    {"id":"SL48","text":"What would make you say 'we're not ready to scale yet'?","sec":"scale","tags":["readiness","risk"],"p":2},
    {"id":"SL49","text":"What would make you confident to add capacity aggressively?","sec":"scale","tags":["readiness","scale"],"p":2},
    {"id":"SL50","text":"What would be catastrophic if loan quality or compliance slipped?","sec":"scale","tags":["risk","compliance"],"p":1},
    {"id":"SL51","text":"What assumptions do you think I might be making that could be wrong?","sec":"vision","tags":["trust","alignment"],"p":2},
    {"id":"SL52","text":"What scares you most about automating underwriting or loan processing?","sec":"vision","tags":["risk","trust"],"p":2},
    {"id":"SL53","text":"What excites you most about automation?","sec":"vision","tags":["motivation"],"p":2},
    {"id":"SL54","text":"What would break trust in this partnership?","sec":"vision","tags":["trust","partnership"],"p":2},
    {"id":"SL55","text":"If this works perfectly, how does your day look different?","sec":"vision","tags":["ideal_state","motivation"],"p":1},
]

BROKER_DEEP = {
    "Intake & Qualification": [
        "Walk through your intake step-by-step from the first message to 'deal created.'",
        "What are the 10 fields you must have to evaluate a deal?",
        "What intake comes via email vs call vs drive link vs form today (percent split)?",
        "Who enters data into your systems today -- you, assistant, no one?",
        "What does an 'ideal intake packet' look like?",
        "What are the minimum docs you require before you'll even review?",
        "How long does it take from 'inquiry received' to 'first response' today?",
        "What % of inquiries never get a response? Why?",
    ],
    "Document Collection & Packaging": [
        "List every document type you require.",
        "How do you request docs today -- email template, form, verbal?",
        "How many times do you typically follow up before getting complete docs?",
        "What % of deals stall permanently at doc collection?",
        "Which doc types take longest to receive?",
        "Who is usually the bottleneck -- borrower, CPA, attorney, seller?",
        "How do you track what's been received vs still needed?",
        "Walk through your packaging process step-by-step from complete docs to lender submission.",
    ],
    "Pipeline Management & Visibility": [
        "What reports do you look at daily/weekly/monthly?",
        "What are the '8am must-know numbers' you want on one dashboard?",
        "What data do you track manually in spreadsheets today?",
        "What would 'perfect pipeline visibility' look like to you?",
        "What forecasting or projection reports would help you?",
        "How do you currently report to stakeholders or partners?",
    ],
    "Lender & Referral Partner Comms": [
        "Walk through a typical lender submission process -- email, portal, phone call?",
        "What information do lenders most frequently come back asking for?",
        "How do you track lender conditions and outstanding requests?",
        "How do you decide when to pull a deal from one lender and resubmit elsewhere?",
        "List your top referral channels -- brokers, MLOs, agents, investors.",
        "What partner behaviors cause the most friction -- late docs, low quality?",
        "How do you communicate with partners today -- email/text/call?",
        "What would make partners send you 2x more deals?",
    ],
    "Closing & Funding Handoffs": [
        "Walk through your closing process from term sheet to funding.",
        "Who coordinates closing -- you, lender, attorney, title company?",
        "What closing documents do you personally prepare vs receive from others?",
        "What typically causes closing delays?",
        "How do you track closing checklist items and who's responsible?",
        "What is the typical time from term sheet to funding?",
    ],
    "Team Accountability & Adoption": [
        "For each role on your team, what are the top 5 tasks they do per deal today?",
        "For each role, what do you want them doing inside the system daily?",
        "Which role is most resistant to new processes historically? Why?",
        "What are the top 3 reasons staff will 'go around' the system?",
        "Where do you expect shadow processes to persist -- texts, personal email, spreadsheets?",
        "How does your team communicate today -- Slack, email, text, meetings?",
    ],
    "Systems Architecture & Source of Truth": [
        "What CRM are you using today? What edition and features do you actually use?",
        "What email provider, and do you use shared inboxes?",
        "What do you use for doc collection, storage, e-sign, texting/calling?",
        "What do you use for task management -- Slack, Asana, Monday, none?",
        "What do you use for underwriting analysis -- Excel, tools, custom templates?",
        "What tools must the new system integrate with on day one?",
        "What systems must talk to each other automatically?",
    ],
    "Reporting & Dashboards": [
        "How do you track expected commission per deal?",
        "What revenue/commission reports do you want but don't have?",
        "What industry regulations apply to your deals -- RESPA, TRID, state licensing?",
        "What audit trail or documentation requirements must you maintain?",
        "How long do you keep deal records and documents?",
        "Do you need role-based permissions -- who can see what?",
    ],
}

LENDER_DEEP = {
    "Application Intake & Screening": [
        "Walk through your intake step-by-step from first submission to 'loan boarded in system.'",
        "What are the required fields/docs for a complete application?",
        "What are the minimum docs you require before underwriting can start?",
        "Who reviews incoming applications first -- processor, junior UW, senior UW?",
        "How long from 'application received' to 'initial response' today?",
        "What % of applications are incomplete at submission?",
        "How do you request missing information?",
        "What does an 'ideal application package' look like?",
    ],
    "Credit Analysis & Underwriting": [
        "Walk through your underwriting process step-by-step.",
        "What credit analysis do you perform -- personal credit, business financials, cash flow, collateral?",
        "What tools do you use for financial spreading and analysis?",
        "What calculations are most time-consuming -- DSCR, debt service, cash flow?",
        "What parts of underwriting require judgment vs are mechanical?",
        "How do you document your credit decision and rationale?",
        "What triggers a decline vs conditional approval vs clear approval?",
        "Who has authority to approve loans at different size/risk thresholds?",
    ],
    "Conditions & Stipulations": [
        "What are the most common conditions you issue -- tax transcripts, insurance, appraisal?",
        "How do you communicate conditions to brokers/borrowers?",
        "How do you track outstanding conditions and which are satisfied?",
        "What % of conditions are 'borrower's issue' vs 'third-party vendor' vs 'internal review'?",
        "How long does condition clearing typically take?",
        "What causes condition delays most often?",
        "How do you know when to escalate or push a stalled loan?",
    ],
    "Compliance & Quality Control": [
        "What regulations govern your lending -- SBA, RESPA, TRID, state licensing?",
        "What compliance checks happen at each loan stage?",
        "Who performs quality control reviews?",
        "What % of loans get QC'd?",
        "What are the most common compliance errors or oversights?",
        "How do you document compliance with regulations?",
        "What audit trail or documentation requirements must you maintain?",
    ],
    "Document Review & Collection": [
        "List every document type you require with examples.",
        "How do you request docs today?",
        "How many times do you typically follow up before getting complete docs?",
        "What % of loans stall at doc collection?",
        "Which doc types take longest to receive?",
        "How do you track what's been received vs still needed?",
        "What doc format issues cause problems -- wrong year, incomplete pages, illegible?",
    ],
    "Closing & Funding": [
        "Walk through your closing process from clear-to-close to funding.",
        "Who coordinates closing -- internal closer, title company, attorney?",
        "What closing documents do you prepare vs receive from others?",
        "What typically causes closing delays?",
        "How do you fund loans -- wire, ACH, check?",
        "What final verifications happen before funding?",
        "What post-closing documentation is required?",
    ],
    "Pipeline & Reporting": [
        "What reports do you look at daily/weekly/monthly?",
        "What are the '8am must-know numbers' you want on one dashboard?",
        "What pipeline visibility do you have today?",
        "What would 'perfect pipeline management' look like?",
        "What forecasting or projection reports would help you?",
        "What data do you track manually in spreadsheets today?",
    ],
    "Systems & Integrations": [
        "What LOS are you using today? What features do you actually use vs unused?",
        "What do you use for credit reports, property valuations, tax return analysis?",
        "What do you use for title and lien searches?",
        "What do you use for pricing and rate sheets?",
        "What systems must talk to each other automatically?",
        "What manual data entry between systems do you want eliminated?",
        "Are there API limitations or costs you're aware of?",
    ],
}

SIGNATURE_QS = [
    {"id":"SIG1","text":"What would break if volume doubled?"},
    {"id":"SIG2","text":"What step do you still personally touch on nearly every file?"},
    {"id":"SIG3","text":"What does your team hate doing most?"},
    {"id":"SIG4","text":"What absolutely cannot happen outside the system?"},
    {"id":"SIG5","text":"What would be catastrophic if automated incorrectly?"},
    {"id":"SIG6","text":"What does a perfect deal or perfect loan look like?"},
]

WORKFLOWS = [
    {"id":"WF1","name":"Smart Document Collection","desc":"Portal + reminders, zero chasing","tags":["document_chasing","follow_up","doc_collection","missing_docs"]},
    {"id":"WF2","name":"Executive Dashboards","desc":"Real-time pipeline & KPI visualization","tags":["no_visibility","manual_reporting","no_source_of_truth","pipeline"]},
    {"id":"WF3","name":"Lead Follow-Up","desc":"5-min first response, multi-channel","tags":["slow_response","lead_followup","intake","conversion"]},
    {"id":"WF4","name":"Onboarding Timeline","desc":"Forms, foldering, timeline tracking","tags":["onboarding","folder_structure","task_tracking","handoff"]},
    {"id":"WF5","name":"Team Accountability","desc":"Daily outreach tracker & leaderboards","tags":["accountability","task_assignment","team_visibility","adoption"]},
    {"id":"WF6","name":"Scheduling & No-Show Recovery","desc":"Confirmations, reminders, auto-rebook","tags":["scheduling","no_shows","calendar"]},
    {"id":"WF7","name":"Pipeline/Deal Tracking","desc":"Automatic stage updates from actions","tags":["pipeline","crm_maintenance","stage_tracking","deal_visibility","no_source_of_truth"]},
]


# ===================================================================
# QUESTION BATCHING ENGINE
# ===================================================================

def get_snapshot_batch(client_type: str, batch_idx: int, answered_keys: set) -> List[Dict[str, Any]]:
    bank = LENDER_SNAPSHOT_QS if client_type == "lender" else BROKER_SNAPSHOT_QS
    remaining = [q for q in bank if q["id"] not in answered_keys]
    if not remaining:
        return []
    start = batch_idx * BATCH_SIZE
    batch = remaining[start:start + BATCH_SIZE]
    if batch_idx > 0:
        used_sigs = {q["id"] for q in SIGNATURE_QS} & answered_keys
        available_sigs = [q for q in SIGNATURE_QS if q["id"] not in used_sigs]
        if available_sigs and batch:
            batch = batch[:BATCH_SIZE - 1] + [available_sigs[0]]
    return batch


def get_deep_questions(client_type: str, module_name: str) -> List[str]:
    bank = LENDER_DEEP if client_type == "lender" else BROKER_DEEP
    return bank.get(module_name, [])


def get_deep_modules(client_type: str) -> List[str]:
    bank = LENDER_DEEP if client_type == "lender" else BROKER_DEEP
    return list(bank.keys())


# ===================================================================
# RAG ENGINE (Item 1: Re-enabled with built-in embeddings)
# ===================================================================

class RAGEngine:
    """RAG using ChromaDB with built-in sentence-transformer embeddings.
    No OpenAI key required -- uses chromadb's default all-MiniLM-L6-v2."""

    def __init__(self):
        self.client = None
        self.embed_fn = None
        self.ready = False

    def initialize(self):
        if not HAS_CHROMA:
            log.info("ChromaDB not installed -- RAG disabled.")
            return
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_DIR)
            # Use ChromaDB's built-in default embedding function (all-MiniLM-L6-v2)
            # No API key needed -- runs locally
            self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
            self.ready = True
            log.info("RAG engine initialized with ChromaDB (local embeddings)")
        except Exception as e:
            log.warning(f"RAG init failed: {e}")

    def ingest_text(self, collection_name: str, text: str, source_name: str = "",
                    chunk_size: int = 800, overlap: int = 100):
        if not self.ready:
            return
        coll = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )
        chunks = self._chunk(text, chunk_size, overlap)
        if not chunks:
            return
        ids = [f"{collection_name}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
        metas = [{"source": source_name or collection_name, "idx": i} for i in range(len(chunks))]
        # ChromaDB add in batches to avoid issues
        for i in range(0, len(chunks), 50):
            batch_chunks = chunks[i:i+50]
            batch_ids = ids[i:i+50]
            batch_metas = metas[i:i+50]
            coll.add(documents=batch_chunks, ids=batch_ids, metadatas=batch_metas)
        log.info(f"Ingested {len(chunks)} chunks into '{collection_name}' from '{source_name}'")

    def ingest_file(self, collection_name: str, filepath: str,
                    chunk_size: int = 800, overlap: int = 100):
        if not self.ready:
            return
        p = Path(filepath)
        if p.suffix == ".docx":
            try:
                import docx
                doc = docx.Document(filepath)
                text = "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
            except ImportError:
                log.warning("python-docx not installed -- skipping .docx ingestion")
                return
        elif p.suffix == ".pdf":
            # Basic text extraction from PDF
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(filepath)
                text = "\n\n".join(page.get_text() for page in doc)
                doc.close()
            except ImportError:
                log.warning("PyMuPDF not installed -- skipping .pdf ingestion")
                return
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
        self.ingest_text(collection_name, text, source_name=p.name, chunk_size=chunk_size, overlap=overlap)

    def query(self, text: str, collections: Optional[List[str]] = None, n: int = 5) -> str:
        if not self.ready:
            return ""
        if not collections:
            collections = ["broker_audit", "lender_audit", "workflows", "sop", "strategy",
                           "scoping_sop"]
        results = []
        for cn in collections:
            try:
                coll = self.client.get_collection(name=cn, embedding_function=self.embed_fn)
                res = coll.query(query_texts=[text], n_results=min(n, 3))
                docs = res.get("documents", [[]])[0]
                dists = res.get("distances", [[]])[0]
                for i, doc in enumerate(docs):
                    results.append({
                        "text": doc,
                        "dist": dists[i] if i < len(dists) else 0,
                        "src": cn,
                    })
            except Exception:
                continue
        results.sort(key=lambda x: x["dist"])
        if not results:
            return ""
        return "\n\n---\n\n".join(f"[{r['src']}] {r['text']}" for r in results[:n])

    def _chunk(self, text: str, size: int, overlap: int) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) > size and current:
                chunks.append(current.strip())
                current = current[-overlap:] + "\n\n" + p if len(current) > overlap else p
            else:
                current += ("\n\n" if current else "") + p
        if current.strip():
            chunks.append(current.strip())
        return chunks


rag = RAGEngine()


# ===================================================================
# PROMPT ENGINE
# ===================================================================

BASE_SYSTEM = """You are Strat AI Solutions' Audit Scoping Bot -- an expert scoping assistant for CRE brokerages, mortgage brokerages, CRE lenders, and adjacent real estate operators. Built by Strat AI Solutions, founded by Yaseen Abdelrahman.

YOUR JOB: Classify -> Qualify -> Snapshot Audit (batched 5-8) -> Synthesize -> Deep Audit (targeted) -> Proposal-Ready Output.

HARD RULES:
- Present questions in a batch (typically 5-8). In the SAME message introducing a batch, tell the user: "Feel free to answer one at a time -- pick whichever you want to start with. Or answer all of them at once if you prefer. Both work." NEVER require all questions to be answered at once. NEVER penalize or re-prompt if the user answers only one, and NEVER penalize if they answer all six at once -- accept any count and proceed.
- Never skip classification or qualification.
- Never recommend automation without explaining the bottleneck first.
- Never hide uncertainty -- state what is missing.
- Never call a poor fit a good fit.
- Never give generic AI ideas -- tie every opportunity to a real workflow, user, system, and bottleneck.
- Push for specifics: volumes, cycle times, team roles, systems, error rates.
- Quantify impact: hours lost, delays, conversion loss, compliance risk.
- If the client is rambling, summarize and redirect.
- Follow bottleneck signals -- pivot when something urgent surfaces.
- Short bullets over long essays.
- After each batch, summarize what you heard, note gaps, then ask the next batch.
- Request artifacts when helpful: SOPs, templates, checklists, pipeline screenshots, email templates.
- NEVER use markdown formatting in your responses. No asterisks for bold, no dashes for lists. Use plain text, numbered lists (1. 2. 3.), and line breaks for readability. No special characters like * or **.

TONE: Founder-friendly. Direct. Analytical. Commercially sharp. Not corporate. Not robotic.

OFFERINGS YOU CAN RECOMMEND:
1. Beta Jumpstart Sprint -- Automate one bottleneck in one week, fixed price, guaranteed result
2. Command Center -- Full workflow buildout across all identified bottlenecks
3. Custom Packages -- Multi-scope bundles, bespoke automation builds
4. Strategic/Flagship Partnership -- Co-building at scale, case study arrangements

BETA WORKFLOW SPRINTS (match to specific bottleneck):
1. Smart Document Collection -- Portal + reminders, zero chasing
2. Executive Dashboards -- Real-time pipeline & KPI visualization
3. Lead Follow-Up -- 5-min first response, multi-channel (email/SMS/voicemail)
4. Onboarding Timeline -- Forms, foldering, timeline tracking
5. Team Accountability -- Daily outreach tracker & leaderboards
6. Scheduling & No-Show Recovery -- Confirmations, reminders, auto-rebook
7. Pipeline/Deal Tracking -- Automatic stage updates from rep and client actions

PATTERN RECOGNITION -- Always watch for:
- No single source of truth
- Document chasing and incomplete submissions
- No proactive status alerts or visibility
- Task assignment and communication overhead
- Business logic in someone's head, not codified
- Reporting that depends on manual updates
- Founder/operator bottleneck -- too much depends on one person

ENGAGEMENT PATH LOGIC:
- One clear bottleneck + existing stack -> Sprint/MVP
- Multiple overlapping issues + no source of truth -> Discovery audit first
- Client knows exact project -> Accelerated scoping
- Client doesn't know where to start -> Structured audit to surface top 3-5 opportunities"""


def build_stage_prompt(session: Session) -> str:
    s = session
    ct = s.client_type.value

    if s.stage == Stage.CLASSIFY:
        return """
CURRENT STAGE: CLASSIFICATION
Determine if this is a Broker or Lender operation.
- Broker = sources, screens, packages, places, or manages financing across lenders/referral partners/borrowers
- Lender = originates, underwrites, approves, conditions, closes, and funds loans internally
- If both, choose dominant workflow first and note secondary

Ask 6 targeted multiple-choice questions. Format EVERY question the same way: one-sentence question on its own line, followed by exactly 4 options labeled "A) ...", "B) ...", "C) ...", "D) ..." -- each option on its own line. No markdown, no bullets, no bold. Use identical spacing across all 6 questions.

Before listing the questions, include this single guidance line verbatim: "Pick one to start with -- we'll work through them in whatever order you prefer. If you'd rather answer all six at once, that also works."

If the user answers only one question, accept it, acknowledge briefly, and keep the remaining questions available. If they answer all six in one message, accept and process without re-prompting. Never require all six up front.

When you have enough info, end your response with the exact text: CLASSIFICATION: BROKER or CLASSIFICATION: LENDER or CLASSIFICATION: HYBRID"""

    elif s.stage == Stage.QUALIFY:
        return f"""
CURRENT STAGE: QUALIFICATION GATE
Client classified as: {ct.upper()}

Start your message with this guidance line verbatim: "Pick one to start -- answer in any order. Or answer all at once if you prefer."

Gather these in a batch of 6-8 questions. Accept any number of answers (one to all). Never penalize or re-prompt for partial responses.
- Decision-maker status (are they the person who signs off?)
- Other stakeholders involved
- Top 1-3 business challenges (specific, not vague)
- Target timeline for implementing a solution
- Budget status (approved, under discussion, not yet?)
- Company size + business model
- Prior AI/automation experience

Flag POOR FIT if: no clear problem, no authority, no budget conversation, wants off-the-shelf SaaS, or unrealistic timeline. Be helpful but label the risk.

When gathered, end with: QUALIFICATION: COMPLETE
If flagging: QUALIFICATION: FLAG - [specific reason]"""

    elif s.stage == Stage.SNAPSHOT:
        answered = set(s.snapshot_answers.keys())
        batch = get_snapshot_batch(ct, s.snapshot_batch, answered)
        q_text = "\n".join(f"- {q['text']}" for q in batch) if batch else "No more questions."
        answered_count = len(s.snapshot_answers)

        sig_text = ""
        if s.snapshot_batch > 0:
            sig_text = """
Also consider weaving in one of these signature questions naturally:
- What would break if volume doubled?
- What does your team hate doing most?
- What would be catastrophic if automated incorrectly?"""

        prev = json.dumps(s.snapshot_answers, indent=1)[:3000]

        return f"""
CURRENT STAGE: SNAPSHOT AUDIT -- Batch {s.snapshot_batch + 1}
Client type: {ct.upper()}
Progress: {answered_count} batches of answers collected so far.

Start your message with this guidance line verbatim: "Pick whichever one you want to start with -- or answer all of them together. Your call."

Ask these questions naturally -- adapt wording to conversation, don't read robotically. Accept any number of answers from the user (one to all) without penalizing or re-prompting:
{q_text}
{sig_text}

After the user responds:
1. Summarize what you heard (short bullets)
2. Flag any bottleneck signals worth following
3. Note any missing specifics (volumes, cycle times, team sizes, systems)

Previous answers collected:
{prev}

When you have gathered at least 3 full batches of substantive answers (covering business overview, volume, bottlenecks, systems, workflows, intake, and docs), end with: SNAPSHOT: COMPLETE
Otherwise end with: SNAPSHOT: CONTINUE"""

    elif s.stage == Stage.SYNTHESIS:
        prev = json.dumps(s.snapshot_answers, indent=1)[:4000]
        qual = json.dumps(s.qual_data, indent=1)[:1000]

        return f"""
CURRENT STAGE: SYNTHESIS
Client type: {ct.upper()}
Qualification data: {qual}
Snapshot answers: {prev}
Flags: {s.flags}

Generate a COMPLETE synthesis with ALL of these sections. Use plain text, numbered lists, and clear headers. NO markdown formatting (no asterisks, no dashes for bullets).

Client Classification
{ct.upper()} -- explain why

Executive Summary
2-3 paragraphs summarizing current operation, team, volume, key workflows

Top 3 Bottlenecks
For each: name, evidence from their answers, estimated impact (hours/week, deals lost, etc.), affected roles

Top 5 Automation Opportunities
For each: opportunity name, which bottleneck it fixes, why it matters NOW, systems involved, effort (low/med/high), expected impact, which Beta Workflow Sprint it maps to

Pattern Flags
Check each and mark Y/N with evidence:
- No single source of truth
- Document chasing
- No proactive alerts
- Task assignment overhead
- Knowledge in heads
- Manual reporting
- Founder bottleneck

Risks & Controls
For each risk: type (operational/compliance/adoption), description, severity, suggested control

Missing Information
What we still don't know and need to find out

Recommended Engagement Path
One of: Sprint, Command Center, Discovery Audit, Accelerated Scoping -- with rationale

30/60/90 Day Plan
0-30: Stabilize, map, quick wins
31-60: Build or pilot highest-ROI workflow
61-90: Expand, enforce adoption, measure KPIs, prep phase 2

Recommended Deep Audit Modules
List 2-3 modules worth deep-diving into based on highest-ROI or highest-risk signals

After presenting the full synthesis, ask the client if they want to deep-dive into specific areas.
End with: SYNTHESIS: COMPLETE"""

    elif s.stage == Stage.DEEP_AUDIT:
        mods = s.deep_modules
        current_mod = mods[s.deep_module_idx] if s.deep_module_idx < len(mods) else "done"
        qs = get_deep_questions(ct, current_mod)
        q_text = "\n".join(f"- {q}" for q in qs[:8])
        prev = json.dumps(s.deep_answers, indent=1)[:2000]

        return f"""
CURRENT STAGE: DEEP AUDIT -- Module: {current_mod} ({s.deep_module_idx + 1} of {len(mods)})
Client type: {ct.upper()}
Remaining modules: {mods[s.deep_module_idx:]}

For this module, ask 5-8 questions to extract:
{q_text}

For each answer, capture: current-state workflow, systems touched, handoffs, manual steps, failure points, time loss, duplicate entry, compliance risks, what must stay human, what could be automated.

Previous deep audit answers: {prev}

When this module is sufficiently covered, end with: DEEP_MODULE: COMPLETE
When ALL modules are done, end with: DEEP_AUDIT: COMPLETE"""

    elif s.stage == Stage.PROPOSAL:
        all_data = {
            "client_type": ct,
            "qualification": s.qual_data,
            "snapshot": s.snapshot_answers,
            "deep_audit": s.deep_answers,
            "flags": s.flags,
        }
        data_str = json.dumps(all_data, indent=1)[:5000]

        return f"""
CURRENT STAGE: PROPOSAL-READY OUTPUT
Convert all audit findings into clear proposal-ready scoping language. Use plain text only, NO markdown.

All collected data:
{data_str}

Synthesis highlights:
{s.synthesis_text[:2000]}

Structure the proposal as:

1. Current State
What exists today -- systems, processes, team, volume, pain points

2. Desired Future State
What the operation should look like post-automation

3. Phase 1 Scope (30-day deliverable)
Highest-ROI build -- specific features, integrations, and workflows
Map to specific Beta Workflow Sprint(s)

4. Phase 2 Opportunities (60-90 day expansion)
Next-priority builds after Phase 1 proves value

5. Assumptions & Dependencies
What must be true for this to work

6. Success Metrics
Specific KPIs: faster intake, shorter doc cycle, reduced manual follow-up, more deals per headcount, fewer missed tasks, better compliance, dashboard visibility

7. Recommended Offering
Sprint ($X range), Command Center ($X range), or Partnership -- with rationale

8. Implementation Sequence
Week-by-week for Phase 1

9. Next Steps
What the client should do right now to get started. Include: book a call at {CALENDLY_URL}

End with: PROPOSAL: COMPLETE"""

    return ""


def build_full_system_prompt(session: Session, rag_context: str = "") -> str:
    parts = [BASE_SYSTEM]
    stage_prompt = build_stage_prompt(session)
    if stage_prompt:
        parts.append(stage_prompt)
    if rag_context:
        parts.append(f"\nRELEVANT KNOWLEDGE BASE CONTEXT:\n{rag_context}")
    return "\n\n".join(parts)


# ===================================================================
# LLM CLIENT (Item 25: Sonnet by default, Item 26: cost tracking)
# ===================================================================

# Sonnet pricing per 1M tokens (as of 2025)
SONNET_INPUT_COST = 3.0 / 1_000_000
SONNET_OUTPUT_COST = 15.0 / 1_000_000


async def call_llm(system_prompt: str, messages: List[Dict[str, Any]],
                   session: Optional[Session] = None) -> str:
    chat_msgs: List[Dict[str, str]] = []
    for m in messages:
        role = "assistant" if m.get("role") == "bot" else "user"
        chat_msgs.append({"role": role, "content": m["content"]})

    # Ensure alternating roles
    deduped: List[Dict[str, str]] = []
    for m in chat_msgs:
        if deduped and deduped[-1]["role"] == m["role"]:
            deduped[-1]["content"] += "\n\n" + m["content"]
        else:
            deduped.append(m)

    if not deduped or deduped[0]["role"] != "user":
        deduped.insert(0, {"role": "user", "content": "Begin."})

    try:
        if not ANTHROPIC_API_KEY:
            return "ERROR: ANTHROPIC_API_KEY is not set. Please add it to your .env file."
        async with httpx.AsyncClient(timeout=90) as client:
            payload = {
                "model": LLM_MODEL,
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": deduped[-MAX_HISTORY:],
            }
            log.info(f"Calling Anthropic API -- model: {LLM_MODEL}, messages: {len(deduped)}")
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            data = resp.json()
            log.info(f"Anthropic response status: {resp.status_code}")
            if resp.status_code != 200:
                log.error(f"Anthropic API error: {json.dumps(data, indent=2)}")
                err_msg = data.get("error", {}).get("message", "Unknown error")
                return f"I encountered an issue connecting to the AI service. Please try again in a moment. (Error: {err_msg})"
            if "content" in data and data["content"]:
                text = data["content"][0]["text"]
                # Item 26: Track cost
                if session:
                    usage = data.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    cost = (input_tokens * SONNET_INPUT_COST) + (output_tokens * SONNET_OUTPUT_COST)
                    session.api_cost_usd += cost
                    session.api_calls += 1
                    log.info(f"Session {session.id[:8]}: +${cost:.4f} (total: ${session.api_cost_usd:.4f})")
                return text
            log.error(f"Unexpected response format: {data}")
            return "I received an unexpected response. Please try sending your message again."
    except httpx.TimeoutException:
        return "The request took too long. Please try again -- I'll work faster this time."
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return f"Connection error -- please try again in a moment. ({type(e).__name__})"


# ===================================================================
# REPORT GENERATION (Item 5: from session data, Item 16: structured)
# ===================================================================

async def generate_structured_report(session: Session) -> str:
    """Generate structured report from stored session data, not live conversation.
    Item 5: Doesn't depend on connection. Item 16: Clean structured format."""

    all_data = {
        "contact": {
            "name": session.contact_name,
            "email": session.contact_email,
            "company": session.company_name,
        },
        "client_type": session.client_type.value,
        "fit_status": session.fit.value,
        "qualification": session.qual_data,
        "snapshot_answers": session.snapshot_answers,
        "deep_audit_answers": session.deep_answers,
        "synthesis": session.synthesis_text[:3000] if session.synthesis_text else "",
        "flags": session.flags,
        "recommended_workflows": [w["workflow"] for w in _match_workflows(session)],
    }

    report_prompt = f"""Generate a structured audit report from the following session data.
This report will be read by a CRE principal -- it must be clear, concise, and actionable.
They should be able to read it in under 5 minutes and know exactly what the problem is,
what the recommendation is, and what the next step is.

STRICT FORMATTING RULES (violating any of these will make the report unpresentable):
- NO markdown: no asterisks (*), no double-asterisks (**), no hashtags (#), no underscores (_), no dashes as bullets
- NO bullet points of any kind. Use numbered lists only.
- Section headers are plain text with no formatting symbols
- Plain text only throughout

Session data:
{json.dumps(all_data, indent=2)[:6000]}

Use this EXACT structure. Section order is LOCKED -- do not reorder, rename, merge, or omit sections. Output sections 1-7 as normal. For section 8 (NEXT STEPS), output ONLY the literal static text provided below -- do not generate dynamic content, do not reword, do not add context. Copy it exactly.

1. CLIENT OVERVIEW
Firm name, type (broker/lender/hybrid), size, key stakeholders, fit status (Good/Moderate/Poor)

2. TOP 3 BOTTLENECKS
Format as a numbered list exactly like this:
1. [Bottleneck Name]
Evidence: [what the audit revealed]
Impact: [estimated time or revenue impact]

2. [Bottleneck Name]
Evidence: [what the audit revealed]
Impact: [estimated time or revenue impact]

3. [Bottleneck Name]
Evidence: [what the audit revealed]
Impact: [estimated time or revenue impact]

3. TOP 3-5 AUTOMATION OPPORTUNITIES
Format as a numbered list. For each:
1. [Workflow Name] (Effort: Low/Medium/High)
Systems: [tools involved]
Outcome: [expected result]

4. RECOMMENDED ENGAGEMENT
Sprint / Command Center / Partnership
One-paragraph rationale

5. PHASE 1 SCOPE
What gets built, on what tools, in what timeframe

6. PHASE 2 OPPORTUNITIES
Follow-on scope items, not in scope for Phase 1

7. SUCCESS METRICS
2-3 quantified KPIs tied to the recommended workflow

8. NEXT STEPS
Schedule a discovery call with Strat AI: {CALENDLY_URL}"""

    report = await call_llm(report_prompt, [{"role": "user", "content": "Generate the report."}])
    report = _strip_report_markdown(report)
    report = _enforce_static_next_steps(report)
    return report


def _strip_report_markdown(text: str) -> str:
    """Remove all markdown artifacts from a generated report so it renders cleanly in PDF/HTML."""
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(.+?)\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_(.+?)_', r'\1', text, flags=re.DOTALL)
    # Remove markdown headers (## Header -> Header, keep section number if any)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove stray leading hashtags (e.g. "# " at line start)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove markdown bullet points
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    # Clean up repeated blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# Section 8 is locked to this exact static CTA -- per spec, do not make dynamic.
STATIC_NEXT_STEPS_BLOCK = (
    "8. NEXT STEPS\n"
    "Schedule a discovery call with Strat AI: {url}"
)


def _enforce_static_next_steps(report: str) -> str:
    """Strip any LLM-generated NEXT STEP(S) section and append the locked CTA."""
    # Remove everything from the NEXT STEPS header onward (with or without the leading "8. ")
    stripped = re.sub(
        r"\n\s*(?:\d\.\s*)?NEXT\s*STEPS?\b.*\Z",
        "",
        report,
        flags=re.IGNORECASE | re.DOTALL,
    ).rstrip()
    cta = STATIC_NEXT_STEPS_BLOCK.format(url=CALENDLY_URL)
    return f"{stripped}\n\n{cta}"


# ===================================================================
# STATE MACHINE & TRANSITION DETECTION
# ===================================================================

CONTROL_PATTERNS = {
    Stage.CLASSIFY: [
        (r"CLASSIFICATION:\s*BROKER", {"client_type": ClientType.BROKER, "next": Stage.QUALIFY}),
        (r"CLASSIFICATION:\s*LENDER", {"client_type": ClientType.LENDER, "next": Stage.QUALIFY}),
        (r"CLASSIFICATION:\s*HYBRID", {"client_type": ClientType.HYBRID, "next": Stage.QUALIFY}),
    ],
    Stage.QUALIFY: [
        (r"QUALIFICATION:\s*COMPLETE", {"next": Stage.SNAPSHOT}),
        (r"QUALIFICATION:\s*FLAG", {"next": Stage.SNAPSHOT, "flag": "qualification_flag"}),
    ],
    Stage.SNAPSHOT: [
        (r"SNAPSHOT:\s*COMPLETE", {"next": Stage.SYNTHESIS}),
        (r"SNAPSHOT:\s*CONTINUE", {"inc_batch": True}),
    ],
    Stage.SYNTHESIS: [
        (r"SYNTHESIS:\s*COMPLETE", {"next": Stage.DEEP_AUDIT}),
    ],
    Stage.DEEP_AUDIT: [
        (r"DEEP_MODULE:\s*COMPLETE", {"inc_module": True}),
        (r"DEEP_AUDIT:\s*COMPLETE", {"next": Stage.PROPOSAL}),
    ],
    Stage.PROPOSAL: [
        (r"PROPOSAL:\s*COMPLETE", {"next": Stage.COMPLETE}),
    ],
}


def detect_transitions(text: str, session: Session) -> Session:
    patterns = CONTROL_PATTERNS.get(session.stage, [])
    for regex, actions in patterns:
        if re.search(regex, text, re.IGNORECASE):
            if "client_type" in actions:
                session.client_type = actions["client_type"]
            if "next" in actions:
                next_stage = actions["next"]
                if next_stage == Stage.DEEP_AUDIT:
                    mods = get_deep_modules(session.client_type.value)
                    session.deep_modules = mods[:3]
                    session.deep_module_idx = 0
                    session.synthesis_text = text
                if next_stage == Stage.COMPLETE:
                    session.completed_at = datetime.utcnow().isoformat() + "Z"
                session.stage = next_stage
            if actions.get("inc_batch"):
                session.snapshot_batch += 1
            if actions.get("inc_module"):
                session.deep_module_idx += 1
                if session.deep_module_idx >= len(session.deep_modules):
                    session.stage = Stage.PROPOSAL
            if "flag" in actions and actions["flag"] not in session.flags:
                session.flags.append(actions["flag"])
            break
    return session


def clean_control_tokens(text: str) -> str:
    patterns = [
        r"CLASSIFICATION:\s*(BROKER|LENDER|HYBRID)\s*",
        r"QUALIFICATION:\s*(COMPLETE|FLAG[^\n]*)\s*",
        r"SNAPSHOT:\s*(COMPLETE|CONTINUE)\s*",
        r"SYNTHESIS:\s*COMPLETE\s*",
        r"DEEP_MODULE:\s*COMPLETE\s*",
        r"DEEP_AUDIT:\s*COMPLETE\s*",
        r"PROPOSAL:\s*COMPLETE\s*",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()


def strip_markdown(text: str) -> str:
    """Item 9: Remove markdown formatting from bot responses."""
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    # Remove markdown headers (## Header -> Header)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Replace n-dashes with regular dashes
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    # Remove markdown bullet points (- item -> item, * item -> item)
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    return text


def extract_user_data(user_msg: str, session: Session) -> None:
    if len(user_msg.strip()) < 10:
        return
    if session.stage == Stage.QUALIFY:
        key = f"qual_{len(session.qual_data)}"
        session.qual_data[key] = user_msg
    elif session.stage == Stage.SNAPSHOT:
        key = f"batch_{session.snapshot_batch}"
        if key in session.snapshot_answers:
            session.snapshot_answers[key] += "\n\n" + user_msg
        else:
            session.snapshot_answers[key] = user_msg
    elif session.stage == Stage.DEEP_AUDIT:
        mod = (
            session.deep_modules[session.deep_module_idx]
            if session.deep_module_idx < len(session.deep_modules)
            else "unknown"
        )
        key = f"{mod}_{len(session.deep_answers)}"
        session.deep_answers[key] = user_msg


# ===================================================================
# SESSION STORE (Item 4: cleanup called properly)
# ===================================================================

class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_or_create(self, sid: str) -> Session:
        if sid not in self._sessions:
            self._sessions[sid] = Session(
                id=sid,
                created_at=datetime.utcnow().isoformat() + "Z",
            )
        return self._sessions[sid]

    def get(self, sid: str) -> Optional[Session]:
        return self._sessions.get(sid)

    def all_sessions(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._sessions.values()]

    def cleanup(self) -> None:
        """Item 4: Actually clean up expired sessions."""
        cutoff = datetime.utcnow() - timedelta(hours=SESSION_TTL_HOURS)
        expired = []
        for sid, s in self._sessions.items():
            try:
                if datetime.fromisoformat(s.created_at) < cutoff:
                    expired.append(sid)
            except Exception:
                expired.append(sid)
        for sid in expired:
            del self._sessions[sid]
        if expired:
            log.info(f"Cleaned up {len(expired)} expired sessions")


store = SessionStore()


# ===================================================================
# SUPABASE PERSISTENCE (Item 3: live admin dashboard)
# ===================================================================

def _session_to_supabase_row(session: Session) -> Dict[str, Any]:
    """Flatten session to a Supabase-safe dict (no Enum objects)."""
    return {
        "id": session.id,
        "created_at": session.created_at,
        "completed_at": session.completed_at or None,
        "stage": session.stage.value if isinstance(session.stage, Enum) else session.stage,
        "client_type": session.client_type.value if isinstance(session.client_type, Enum) else session.client_type,
        "fit": session.fit.value if isinstance(session.fit, Enum) else session.fit,
        "contact_name": session.contact_name,
        "contact_email": session.contact_email,
        "company_name": session.company_name,
        "api_cost_usd": round(session.api_cost_usd, 6),
        "api_calls": session.api_calls,
        "calendly_clicked": session.calendly_clicked,
        "flags": session.flags,
        "snapshot_batch": session.snapshot_batch,
        "snapshot_answers": session.snapshot_answers,
        "deep_answers": session.deep_answers,
        "qual_data": session.qual_data,
        "synthesis_text": session.synthesis_text[:4000] if session.synthesis_text else "",
        "deep_modules": session.deep_modules,
        "deep_module_idx": session.deep_module_idx,
        "metadata": session.metadata,
    }


async def save_to_supabase(session: Session) -> None:
    """Upsert session row into Supabase. No-op if not configured."""
    if not _supabase_client:
        return
    try:
        row = _session_to_supabase_row(session)
        _supabase_client.table("sessions").upsert(row, on_conflict="id").execute()
    except Exception as e:
        log.warning(f"Supabase save failed for session {session.id}: {e}")


async def load_supabase_sessions() -> List[Dict[str, Any]]:
    """Fetch all sessions from Supabase. Returns [] if not configured."""
    if not _supabase_client:
        return []
    try:
        result = _supabase_client.table("sessions").select("*").order("created_at", desc=True).execute()
        return result.data or []
    except Exception as e:
        log.warning(f"Supabase load failed: {e}")
        return []


# ===================================================================
# INTEGRATIONS (Items 18, 23)
# ===================================================================

async def push_to_hubspot(session: Session) -> None:
    """Item 18: Push completed session to HubSpot CRM."""
    if not HUBSPOT_API_KEY:
        log.info("HubSpot integration not configured -- skipping CRM push")
        return
    try:
        workflows = _match_workflows(session)
        top_bottleneck = ""
        if session.flags:
            top_bottleneck = session.flags[0]

        contact_data = {
            "properties": {
                "email": session.contact_email,
                "firstname": session.contact_name.split()[0] if session.contact_name else "",
                "lastname": " ".join(session.contact_name.split()[1:]) if session.contact_name else "",
                "company": session.company_name,
                "hs_lead_status": "NEW",
                "strat_ai_fit_status": session.fit.value,
                "strat_ai_client_type": session.client_type.value,
                "strat_ai_top_bottleneck": top_bottleneck,
                "strat_ai_recommended_engagement": workflows[0]["workflow"] if workflows else "",
                "strat_ai_session_id": session.id,
            }
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.hubapi.com/crm/v3/objects/contacts",
                headers={
                    "Authorization": f"Bearer {HUBSPOT_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=contact_data,
            )
            if resp.status_code in (200, 201):
                log.info(f"HubSpot contact created for {session.contact_email}")
            else:
                log.warning(f"HubSpot push failed: {resp.status_code} {resp.text}")
    except Exception as e:
        log.error(f"HubSpot integration error: {e}")


async def send_slack_notification(session: Session) -> None:
    """Item 23: Slack notification on high-fit completion."""
    if not SLACK_WEBHOOK_URL:
        log.info("Slack integration not configured -- skipping notification")
        return
    if session.fit != Fit.GOOD:
        return
    try:
        workflows = _match_workflows(session)
        message = {
            "text": (
                f"HIGH-FIT PROSPECT COMPLETED AUDIT\n"
                f"Company: {session.company_name}\n"
                f"Contact: {session.contact_name} ({session.contact_email})\n"
                f"Type: {session.client_type.value.upper()}\n"
                f"Top Bottleneck: {session.flags[0] if session.flags else 'N/A'}\n"
                f"Recommended: {workflows[0]['workflow'] if workflows else 'N/A'}\n"
                f"Session Cost: ${session.api_cost_usd:.2f}"
            )
        }
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(SLACK_WEBHOOK_URL, json=message)
            log.info(f"Slack notification sent for {session.company_name}")
    except Exception as e:
        log.error(f"Slack notification error: {e}")


# ===================================================================
# CONVERSATION HANDLER
# ===================================================================

async def handle_message(session: Session, user_message: str) -> Dict[str, Any]:
    extract_user_data(user_message, session)

    session.messages.append({
        "role": "user",
        "content": user_message,
        "ts": datetime.utcnow().isoformat(),
    })

    # Item 1: RAG context retrieval
    rag_context = ""
    if rag.ready:
        colls_map = {
            Stage.CLASSIFY: ["broker_audit", "lender_audit"],
            Stage.QUALIFY: ["sop", "strategy", "scoping_sop"],
            Stage.SNAPSHOT: ["broker_audit" if session.client_type != ClientType.LENDER else "lender_audit"],
            Stage.SYNTHESIS: ["workflows", "strategy"],
            Stage.DEEP_AUDIT: ["broker_audit" if session.client_type != ClientType.LENDER else "lender_audit", "sop"],
            Stage.PROPOSAL: ["workflows", "sop", "strategy", "scoping_sop"],
        }
        target = colls_map.get(session.stage, ["strategy"])
        rag_context = rag.query(user_message, collections=target, n=4)

    system_prompt = build_full_system_prompt(session, rag_context)
    response = await call_llm(system_prompt, session.messages[-MAX_HISTORY:], session=session)

    prev_stage = session.stage
    session = detect_transitions(response, session)
    display_text = clean_control_tokens(response)
    display_text = strip_markdown(display_text)  # Item 9

    session.messages.append({
        "role": "bot",
        "content": display_text,
        "ts": datetime.utcnow().isoformat(),
    })

    # Item 19: Add Calendly link at proposal/complete stage
    calendly_show = session.stage in (Stage.PROPOSAL, Stage.COMPLETE)

    # If just completed, fire integrations
    if session.stage == Stage.COMPLETE and prev_stage != Stage.COMPLETE:
        asyncio.create_task(push_to_hubspot(session))
        asyncio.create_task(send_slack_notification(session))

    # Persist to Supabase on every stage transition or stage-change
    if session.stage != prev_stage or session.stage in (Stage.PROPOSAL, Stage.COMPLETE):
        asyncio.create_task(save_to_supabase(session))

    return {
        "type": "bot_message",
        "content": display_text,
        "stage": session.stage.value,
        "stage_label": STAGE_LABELS.get(session.stage, ""),
        "client_type": session.client_type.value,
        "fit": session.fit.value,
        "flags": session.flags,
        "progress": session.progress_pct(),
        "snapshot_count": len(session.snapshot_answers),
        "snapshot_batch": session.snapshot_batch,
        "show_calendly": calendly_show,
        "calendly_url": CALENDLY_URL if calendly_show else "",
        "api_cost": round(session.api_cost_usd, 4),
        "summary": summarize_progress(session),
    }


# ===================================================================
# WORKFLOW MATCHING
# ===================================================================

# Human-readable labels for the tag signals used in _match_workflows.
BOTTLENECK_LABELS = {
    "document_chasing": "Document chasing",
    "follow_up": "Manual follow-ups",
    "no_source_of_truth": "No single source of truth",
    "pipeline": "Pipeline visibility gaps",
    "manual_reporting": "Manual reporting",
    "intake": "Slow / inconsistent intake",
    "adoption": "Team adoption / shadow processes",
    "bottleneck": "Operational bottleneck flagged",
    "founder_bottleneck": "Founder / operator bottleneck",
    "automation": "Repetitive manual work",
    "compliance": "Compliance / audit-trail risk",
    "task_assignment": "Task assignment overhead",
    "scheduling": "Scheduling / no-shows",
}


def summarize_progress(session: Session) -> Dict[str, Any]:
    """Build the live 'Summary So Far' payload for the interactive dashboard panel."""
    matched = _match_workflows(session)
    signals: set = set()
    for m in matched:
        for sig in m.get("matched_signals", []):
            signals.add(sig)
    bottlenecks = [BOTTLENECK_LABELS.get(s, s.replace("_", " ").title()) for s in signals]
    bottlenecks = bottlenecks[:6]
    opportunities = [
        {"name": m["workflow"], "desc": m["description"], "score": m["match_score"]}
        for m in matched[:6]
    ]

    # Count substantive user answers across the audit
    questions_answered = 0
    questions_answered += len(session.qual_data)
    for v in session.snapshot_answers.values():
        if isinstance(v, str):
            # crude split on double newline = per-question answer chunk
            questions_answered += max(1, len([b for b in v.split("\n\n") if b.strip()]))
    questions_answered += len(session.deep_answers)

    # Visible stages -- 6 substantive stages (exclude INTAKE, COMPLETE)
    stage_sequence = [
        Stage.CLASSIFY, Stage.QUALIFY, Stage.SNAPSHOT,
        Stage.SYNTHESIS, Stage.DEEP_AUDIT, Stage.PROPOSAL,
    ]
    current_idx = stage_sequence.index(session.stage) if session.stage in stage_sequence else (
        len(stage_sequence) if session.stage == Stage.COMPLETE else -1
    )
    stages_list = []
    for i, st in enumerate(stage_sequence):
        if session.stage == Stage.COMPLETE:
            status = "done"
        elif i < current_idx:
            status = "done"
        elif i == current_idx:
            status = "active"
        else:
            status = "upcoming"
        stages_list.append({
            "key": st.value,
            "label": STAGE_LABELS.get(st, st.value),
            "status": status,
        })
    stages_completed_count = sum(1 for s in stages_list if s["status"] == "done")

    # Session duration in minutes
    duration_min = 0
    try:
        started = datetime.fromisoformat(session.created_at)
        duration_min = max(0, int((datetime.utcnow() - started).total_seconds() // 60))
    except Exception:
        pass

    fit_labels = {"good": "Good Fit", "moderate": "Moderate Fit", "poor": "Poor Fit", "unknown": "Assessing"}
    type_labels = {"broker": "Broker", "lender": "Lender", "hybrid": "Hybrid", "unknown": "Classifying"}

    return {
        "company": session.company_name or "",
        "contact_name": session.contact_name or "",
        "contact_email": session.contact_email or "",
        "stage": session.stage.value,
        "stage_label": STAGE_LABELS.get(session.stage, ""),
        "progress": session.progress_pct(),
        "client_type": session.client_type.value,
        "client_type_label": type_labels.get(session.client_type.value, "Classifying"),
        "fit": session.fit.value,
        "fit_label": fit_labels.get(session.fit.value, "Assessing"),
        "stages": stages_list,
        "stages_completed": stages_completed_count,
        "stages_total": len(stage_sequence),
        "questions_answered": questions_answered,
        "duration_min": duration_min,
        "bottlenecks": bottlenecks,
        "bottleneck_count": len(bottlenecks),
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),
        "flags": session.flags,
        "contact": {
            "website": STRAT_AI_WEBSITE_URL,
            "email": STRAT_AI_CONTACT_EMAIL,
        },
    }


def _match_workflows(session: Session) -> List[Dict[str, Any]]:
    all_tags: set = set()
    for answers in [session.snapshot_answers, session.deep_answers]:
        for v in answers.values():
            if not isinstance(v, str):
                continue
            text = v.lower()
            tag_signals = {
                "document_chasing": ["missing doc", "chase", "follow up on doc", "never get", "incomplete"],
                "follow_up": ["follow up", "reminder", "chase", "nag"],
                "no_source_of_truth": ["no single", "scattered", "multiple systems", "no source of truth", "spreadsheet"],
                "pipeline": ["pipeline", "deal stage", "where things are", "visibility"],
                "manual_reporting": ["manual report", "spreadsheet", "pull numbers", "track manually"],
                "intake": ["intake", "first response", "come in via", "new deal", "new loan"],
                "adoption": ["resist", "won't use", "go around", "shadow", "habit"],
                "bottleneck": ["bottleneck", "stuck", "slow", "delay", "break"],
                "founder_bottleneck": ["personally", "touch every", "in my head", "only I know"],
                "automation": ["repetitive", "always the same", "copy paste", "manual"],
                "compliance": ["compliance", "audit trail", "regulation", "risk"],
                "task_assignment": ["assign", "who's doing", "delegate", "coordinate"],
                "scheduling": ["schedule", "no-show", "calendar", "meeting"],
            }
            for tag, keywords in tag_signals.items():
                if any(kw in text for kw in keywords):
                    all_tags.add(tag)

    matched = []
    for wf in WORKFLOWS:
        overlap = set(wf["tags"]) & all_tags
        if overlap:
            matched.append({
                "workflow": wf["name"],
                "description": wf["desc"],
                "match_score": len(overlap),
                "matched_signals": list(overlap),
            })
    matched.sort(key=lambda x: x["match_score"], reverse=True)
    return matched


# ===================================================================
# FASTAPI APPLICATION
# ===================================================================

app = FastAPI(title="Strat AI Audit & Scoping Bot", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    # Item 1: Initialize RAG
    rag.initialize()
    data_path = Path(DATA_DIR)
    if data_path.exists():
        # Item 12-15: Ingest all knowledge base documents
        mapping = {
            "broker-audit": "broker_audit",
            "lender-audit": "lender_audit",
            "workflow": "workflows",
            "sop": "sop",
            "strategy": "strategy",
            "scoping_sop": "scoping_sop",
        }
        for f in data_path.glob("*"):
            if f.suffix in (".txt", ".docx", ".md"):
                coll_name = "general"
                for key, val in mapping.items():
                    if key in f.stem.lower():
                        coll_name = val
                        break
                try:
                    rag.ingest_file(coll_name, str(f))
                except Exception as e:
                    log.warning(f"Failed to ingest {f}: {e}")
    # Item 4: Start cleanup loop (was defined but task created correctly)
    asyncio.create_task(_cleanup_loop())
    # Load any prior sessions from Supabase into the in-memory store
    if _supabase_client:
        try:
            prior = await load_supabase_sessions()
            for row in prior:
                sid = row.get("id")
                if sid and sid not in store._sessions:
                    s = Session(
                        id=sid,
                        created_at=row.get("created_at", ""),
                        completed_at=row.get("completed_at", ""),
                        stage=Stage(row["stage"]) if row.get("stage") else Stage.INTAKE,
                        client_type=ClientType(row["client_type"]) if row.get("client_type") else ClientType.UNKNOWN,
                        fit=Fit(row["fit"]) if row.get("fit") else Fit.UNKNOWN,
                        contact_name=row.get("contact_name", ""),
                        contact_email=row.get("contact_email", ""),
                        company_name=row.get("company_name", ""),
                        api_cost_usd=float(row.get("api_cost_usd") or 0),
                        api_calls=int(row.get("api_calls") or 0),
                        calendly_clicked=bool(row.get("calendly_clicked")),
                        flags=row.get("flags") or [],
                        snapshot_batch=int(row.get("snapshot_batch") or 0),
                        snapshot_answers=row.get("snapshot_answers") or {},
                        deep_answers=row.get("deep_answers") or {},
                        qual_data=row.get("qual_data") or {},
                        synthesis_text=row.get("synthesis_text") or "",
                        deep_modules=row.get("deep_modules") or [],
                        deep_module_idx=int(row.get("deep_module_idx") or 0),
                        metadata=row.get("metadata") or {},
                    )
                    store._sessions[sid] = s
            log.info(f"Loaded {len(prior)} sessions from Supabase")
        except Exception as e:
            log.warning(f"Failed to load Supabase sessions on startup: {e}")
    log.info("Server ready")


async def _cleanup_loop():
    """Item 4: Properly running cleanup loop."""
    while True:
        try:
            await asyncio.sleep(3600)
            store.cleanup()
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning(f"Cleanup loop error: {e}")


# Serve static files (logo, etc.)
frontend_path = Path(FRONTEND_DIR)
if not frontend_path.exists():
    frontend_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ===================================================================
# FRONTEND HTML (Items 2,3,6,7,8,9,10,11,17,19 — all UI/UX fixes)
# ===================================================================

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Strat AI — Audit & Scoping Bot</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:radial-gradient(ellipse at top,#0a1428 0%,#060b18 60%);color:#e0e0e0;height:100vh;display:flex;justify-content:center;overflow:hidden}
#layout{display:flex;width:100%;max-width:1280px;height:100vh}
#app{flex:1;min-width:0;max-width:820px;height:100vh;display:flex;flex-direction:column;background:#0a1020;position:relative}

/* Left dashboard panel -- hidden until audit begins */
#side-panel{width:340px;flex-shrink:0;background:linear-gradient(180deg,#0a1428 0%,#080e1e 100%);border-right:1px solid #1a2540;display:none;flex-direction:column;height:100vh;box-shadow:inset -1px 0 0 rgba(91,184,245,0.05)}
#side-panel.visible{display:flex;animation:slideInLeft .45s cubic-bezier(.2,.7,.2,1)}
@keyframes slideInLeft{from{opacity:0;transform:translateX(-12px)}to{opacity:1;transform:translateX(0)}}

.sp-brandbar{display:flex;align-items:center;gap:10px;padding:18px 20px 14px;border-bottom:1px solid rgba(26,42,80,.6)}
.sp-brandbar img{width:30px;height:30px;border-radius:8px;object-fit:contain}
.sp-brandbar .sp-brand-text{font-size:11px;font-weight:700;color:#5bb8f5;letter-spacing:.8px;text-transform:uppercase}

.sp-client-card{padding:16px 20px;border-bottom:1px solid rgba(26,42,80,.6);background:linear-gradient(135deg,rgba(26,111,181,.08),transparent)}
.sp-client-card .sp-company{font-size:15px;font-weight:700;color:#fff;letter-spacing:-.2px;margin-bottom:2px;word-break:break-word}
.sp-client-card .sp-contact{font-size:11.5px;color:#8899bb;font-weight:500}
.sp-badges{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}
.sp-pill{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:20px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.sp-pill-type{background:rgba(91,184,245,.12);color:#5bb8f5;border:1px solid rgba(91,184,245,.25)}
.sp-pill-fit-good{background:rgba(74,222,128,.12);color:#4ade80;border:1px solid rgba(74,222,128,.3)}
.sp-pill-fit-moderate{background:rgba(251,191,36,.12);color:#fbbf24;border:1px solid rgba(251,191,36,.3)}
.sp-pill-fit-poor{background:rgba(248,113,113,.12);color:#f87171;border:1px solid rgba(248,113,113,.3)}
.sp-pill-fit-unknown{background:rgba(102,136,170,.12);color:#8899bb;border:1px solid rgba(102,136,170,.25)}

.sp-content{flex:1;overflow-y:auto;padding:18px 20px;display:flex;flex-direction:column;gap:18px}
.sp-section{display:flex;flex-direction:column;gap:8px}
.sp-section-title{display:flex;align-items:center;justify-content:space-between;font-size:10px;font-weight:700;color:#6688aa;text-transform:uppercase;letter-spacing:.7px}
.sp-count{display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:18px;padding:0 6px;background:rgba(91,184,245,.15);color:#5bb8f5;font-size:10px;font-weight:700;border-radius:20px;border:1px solid rgba(91,184,245,.3)}

/* Stat grid 2x2 */
.sp-stats{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.sp-stat{padding:12px;background:rgba(15,24,48,.6);border:1px solid rgba(26,42,80,.8);border-radius:10px;transition:all .2s}
.sp-stat:hover{border-color:rgba(91,184,245,.4);background:rgba(15,24,48,.85);transform:translateY(-1px)}
.sp-stat .sp-stat-value{font-size:22px;font-weight:700;color:#fff;line-height:1;letter-spacing:-.5px}
.sp-stat .sp-stat-label{font-size:10px;color:#8899bb;margin-top:4px;text-transform:uppercase;letter-spacing:.4px;font-weight:500}

/* Progress ring */
.sp-progress-wrap{display:flex;align-items:center;gap:14px;padding:14px;background:rgba(15,24,48,.6);border:1px solid rgba(26,42,80,.8);border-radius:10px}
.sp-ring{position:relative;width:64px;height:64px;flex-shrink:0}
.sp-ring svg{width:100%;height:100%;transform:rotate(-90deg)}
.sp-ring-bg{stroke:rgba(26,42,80,.8);fill:none;stroke-width:6}
.sp-ring-fg{stroke:url(#spGrad);fill:none;stroke-width:6;stroke-linecap:round;transition:stroke-dashoffset .6s cubic-bezier(.2,.7,.2,1)}
.sp-ring-text{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#fff;letter-spacing:-.3px}
.sp-progress-meta{flex:1;min-width:0}
.sp-progress-meta .sp-current-stage{font-size:12.5px;font-weight:700;color:#5bb8f5;margin-bottom:2px}
.sp-progress-meta .sp-stage-sub{font-size:10.5px;color:#8899bb}

/* Stage pipeline */
.sp-pipeline{display:flex;flex-direction:column;gap:4px}
.sp-stage-row{display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:8px;border:1px solid transparent;transition:all .2s}
.sp-stage-row.active{background:rgba(91,184,245,.08);border-color:rgba(91,184,245,.25)}
.sp-stage-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;border:2px solid;background:transparent;transition:all .3s}
.sp-stage-row.done .sp-stage-dot{background:#4ade80;border-color:#4ade80;box-shadow:0 0 8px rgba(74,222,128,.4)}
.sp-stage-row.active .sp-stage-dot{background:#5bb8f5;border-color:#5bb8f5;animation:sp-pulse 1.6s ease-in-out infinite}
.sp-stage-row.upcoming .sp-stage-dot{border-color:#1a2a50}
@keyframes sp-pulse{0%,100%{box-shadow:0 0 0 0 rgba(91,184,245,.6)}50%{box-shadow:0 0 0 6px rgba(91,184,245,0)}}
.sp-stage-name{font-size:12px;font-weight:500;flex:1}
.sp-stage-row.done .sp-stage-name{color:#9fb8d4}
.sp-stage-row.active .sp-stage-name{color:#fff;font-weight:600}
.sp-stage-row.upcoming .sp-stage-name{color:#445a7a}
.sp-stage-check{font-size:11px;color:#4ade80;font-weight:700}
.sp-stage-row:not(.done) .sp-stage-check{display:none}

/* Lists */
.sp-list{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:6px}
.sp-list li{padding:10px 12px;background:rgba(15,24,48,.6);border:1px solid rgba(26,42,80,.8);border-radius:8px;font-size:12px;color:#d8e8ff;line-height:1.45;transition:all .2s;cursor:default;position:relative;overflow:hidden}
.sp-list li:hover{border-color:rgba(91,184,245,.4);background:rgba(15,24,48,.9);transform:translateX(2px)}
.sp-list li::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;background:linear-gradient(180deg,#5bb8f5,#1a6fb5);opacity:0;transition:opacity .2s}
.sp-list li:hover::before{opacity:1}
.sp-list li .opp-name{display:block;font-weight:700;color:#5bb8f5;margin-bottom:3px;font-size:12px;letter-spacing:-.1px}
.sp-list li .opp-desc{font-size:11px;color:#8899bb;line-height:1.45}
.sp-empty{font-size:11.5px;color:#445a7a;font-style:italic;padding:10px 12px;text-align:center;border:1px dashed rgba(26,42,80,.8);border-radius:8px}

.sp-footer{padding:16px 20px;border-top:1px solid rgba(26,42,80,.6);flex-shrink:0;background:rgba(6,11,24,.6)}
.sp-footer .sp-brand{font-size:12px;font-weight:700;color:#d8e8ff;margin-bottom:2px;letter-spacing:-.1px}
.sp-footer .sp-tagline{font-size:10.5px;color:#6688aa;margin-bottom:10px;letter-spacing:.2px}
.sp-footer .sp-link{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:9px 12px;background:linear-gradient(135deg,rgba(26,111,181,.15),rgba(91,184,245,.08));border:1px solid rgba(91,184,245,.3);border-radius:8px;color:#5bb8f5;text-decoration:none;font-size:11.5px;font-weight:600;transition:all .2s}
.sp-footer .sp-link:hover{background:linear-gradient(135deg,rgba(26,111,181,.3),rgba(91,184,245,.15));border-color:rgba(91,184,245,.5);color:#fff;transform:translateY(-1px)}
.sp-footer .sp-link .sp-link-arrow{transition:transform .2s}
.sp-footer .sp-link:hover .sp-link-arrow{transform:translateX(2px)}

@media(max-width:960px){
  #side-panel{position:absolute;left:0;top:0;z-index:50;width:300px;box-shadow:4px 0 24px rgba(0,0,0,.6)}
  #layout{max-width:100%}
}
@media(max-width:700px){
  #side-panel.visible{display:none}
}

/* Custom scrollbar for panel */
.sp-content::-webkit-scrollbar{width:4px}
.sp-content::-webkit-scrollbar-track{background:transparent}
.sp-content::-webkit-scrollbar-thumb{background:rgba(91,184,245,.2);border-radius:2px}

#header{display:flex;justify-content:space-between;align-items:center;padding:14px 22px;border-bottom:1px solid rgba(26,42,80,.6);background:linear-gradient(180deg,#0a1428,#080e1e);flex-shrink:0;backdrop-filter:blur(8px)}
.logo{display:flex;align-items:center;gap:12px}
.logo img{width:36px;height:36px;border-radius:9px;object-fit:contain}
.logo-fallback{width:36px;height:36px;background:linear-gradient(135deg,#0a4a7a,#1e90ff);border-radius:9px;flex-shrink:0}
#header-right{display:flex;align-items:center;gap:10px}
#export-btn{display:none;padding:6px 14px;background:transparent;border:1px solid rgba(91,184,245,.4);color:#5bb8f5;border-radius:20px;font-size:11px;font-weight:600;cursor:pointer;letter-spacing:.3px;font-family:inherit;transition:all .2s}
#export-btn:hover{background:rgba(26,111,181,.15);border-color:rgba(91,184,245,.7);color:#fff}
#badge{font-size:10.5px;padding:5px 12px;border-radius:20px;background:rgba(26,111,181,.15);color:#5bb8f5;font-weight:700;text-transform:uppercase;letter-spacing:.6px;border:1px solid rgba(91,184,245,.25)}
#progress-bar{height:2px;background:rgba(17,24,40,.6);flex-shrink:0;overflow:hidden}
#progress-fill{height:100%;width:0%;background:linear-gradient(90deg,#0a4a7a,#1e90ff,#5bb8f5);transition:width .6s cubic-bezier(.2,.7,.2,1);box-shadow:0 0 12px rgba(91,184,245,.5)}
#messages{flex:1;overflow-y:auto;padding:24px 22px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:84%;padding:14px 18px;border-radius:14px;line-height:1.65;font-size:13.5px;white-space:pre-wrap;word-break:break-word;animation:msgIn .35s cubic-bezier(.2,.7,.2,1);box-shadow:0 2px 10px rgba(0,0,0,.15)}
@keyframes msgIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.msg.bot{align-self:flex-start;background:linear-gradient(135deg,#0f1830,#0c1528);border:1px solid rgba(26,42,80,.8);color:#d0d8e8}
.msg.user{align-self:flex-end;background:linear-gradient(135deg,#0a1e3d,#0c2550);border:1px solid rgba(26,53,112,.8);color:#d8e8ff}
.msg-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px;opacity:.9}
.msg.bot .msg-label{color:#5bb8f5}
.msg.user .msg-label{color:#60a5fa}

/* Question batch progress chip */
.stage-chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;background:rgba(91,184,245,.1);color:#5bb8f5;border:1px solid rgba(91,184,245,.25);margin-bottom:10px}
.stage-chip .chip-dot{width:6px;height:6px;border-radius:50%;background:#5bb8f5;flex-shrink:0}

/* Standardized MCQ styling -- identical container, spacing, typography across all questions */
.mcq-block{margin:14px 0;padding:14px;background:#0b1428;border:1px solid #1a2a50;border-radius:10px}
.mcq-question{font-size:13.5px;font-weight:600;color:#d8e8ff;line-height:1.55;margin-bottom:10px}
.mcq-options{display:flex;flex-direction:column;gap:8px}
.mcq-btn{display:block;width:100%;padding:10px 14px;background:#0a2a4a;border:1px solid #1a6fb5;color:#5bb8f5;border-radius:8px;cursor:pointer;font-size:13px;font-family:inherit;line-height:1.45;transition:all .15s;text-align:left;position:relative}
.mcq-btn:hover{background:#1a6fb5;color:#fff;border-color:#5bb8f5}
.mcq-btn.selected{background:#1a6fb5;color:#fff;border-color:#5bb8f5}
.mcq-btn.selected::after{content:' \2713';font-weight:700}
.mcq-btn:disabled{opacity:.4;cursor:not-allowed;pointer-events:none}

/* Calendly banner -- polished */
.calendly-banner{margin:14px 0;padding:18px 20px;background:linear-gradient(135deg,rgba(26,111,181,.15),rgba(91,184,245,.08));border:1px solid rgba(91,184,245,.35);border-radius:14px;text-align:center;box-shadow:0 8px 24px rgba(26,111,181,.15)}
.calendly-banner a{display:inline-block;color:#5bb8f5;font-weight:700;text-decoration:none;font-size:14.5px;letter-spacing:.2px;padding:4px 0;transition:color .2s}
.calendly-banner a:hover{color:#fff}
.calendly-banner p{color:#8899bb;font-size:12px;margin-top:6px;line-height:1.5}

#input-area{display:flex;gap:10px;padding:16px 22px;border-top:1px solid rgba(26,42,80,.6);background:linear-gradient(180deg,#080e1e,#060b18);flex-shrink:0;align-items:flex-end}
#msg-input{flex:1;background:#0f1830;border:1px solid rgba(26,42,80,.8);border-radius:12px;color:#e0e0e0;padding:12px 16px;font-size:13.5px;font-family:inherit;resize:none;outline:none;line-height:1.55;transition:border-color .2s,box-shadow .2s}
#msg-input:focus{border-color:#5bb8f5;box-shadow:0 0 0 3px rgba(91,184,245,.15)}
#send-btn{padding:12px 26px;background:linear-gradient(135deg,#1a6fb5,#1e90ff);color:#fff;border:none;border-radius:12px;font-weight:700;cursor:pointer;font-size:13.5px;flex-shrink:0;transition:all .2s;box-shadow:0 4px 14px rgba(26,111,181,.35);letter-spacing:.3px}
#send-btn:hover{background:linear-gradient(135deg,#1e90ff,#5bb8f5);transform:translateY(-1px);box-shadow:0 6px 18px rgba(91,184,245,.4)}
#send-btn:disabled{background:#1a2540;cursor:not-allowed;box-shadow:none;transform:none;color:#445a7a}
#info-bar{padding:8px 22px;border-top:1px solid rgba(17,24,40,.6);background:#060b18;display:flex;justify-content:space-between;font-size:10.5px;color:#445a7a;flex-shrink:0;letter-spacing:.3px}
#typing{color:#5bb8f5;font-style:italic;font-size:12px;padding:6px 0;display:flex;align-items:center;gap:6px}
.dot{width:5px;height:5px;border-radius:50%;background:#5bb8f5;animation:pulse 1.2s ease-in-out infinite}
.dot:nth-child(2){animation-delay:.2s}
.dot:nth-child(3){animation-delay:.4s}
@keyframes pulse{0%,100%{opacity:.3;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}

/* Intake form -- polished modal */
#intake-overlay{position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(6,11,24,0.92);backdrop-filter:blur(6px);display:flex;align-items:center;justify-content:center;z-index:100;animation:fadeIn .3s}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
#intake-form{background:linear-gradient(145deg,#0f1830,#0a1428);border:1px solid rgba(26,42,80,.8);border-radius:20px;padding:40px 36px;width:90%;max-width:440px;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.6),0 0 0 1px rgba(91,184,245,.05)}
#intake-form h2{color:#fff;font-size:22px;margin-bottom:8px;letter-spacing:-.3px;font-weight:700}
#intake-form p{color:#8899bb;font-size:13.5px;margin-bottom:22px;line-height:1.6}
#intake-form input{width:100%;padding:13px 16px;margin-bottom:12px;background:rgba(10,16,32,.8);border:1px solid rgba(26,42,80,.8);border-radius:10px;color:#e0e0e0;font-size:14px;font-family:inherit;outline:none;transition:all .2s}
#intake-form input:focus{border-color:#5bb8f5;box-shadow:0 0 0 3px rgba(91,184,245,.15);background:rgba(10,16,32,1)}
#intake-form input::placeholder{color:#445a7a}
#intake-submit{width:100%;padding:14px;background:linear-gradient(135deg,#1a6fb5,#1e90ff);color:#fff;border:none;border-radius:12px;font-weight:700;font-size:15px;cursor:pointer;margin-top:6px;transition:all .25s;box-shadow:0 6px 20px rgba(26,111,181,.35);letter-spacing:.3px}
#intake-submit:hover{background:linear-gradient(135deg,#1e90ff,#5bb8f5);transform:translateY(-1px);box-shadow:0 10px 28px rgba(91,184,245,.45)}
#intake-submit:disabled{background:#1a2540;cursor:not-allowed;box-shadow:none;transform:none}
.intake-error{color:#f87171;font-size:12px;margin-bottom:10px;display:none;padding:8px 12px;background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.25);border-radius:8px}

#welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:22px;text-align:center;padding:48px 40px}
#welcome img{width:84px;height:84px;border-radius:22px;object-fit:contain;box-shadow:0 10px 40px rgba(26,111,181,.3)}
#welcome .logo-fallback-big{width:84px;height:84px;background:linear-gradient(135deg,#0a4a7a,#1e90ff);border-radius:22px;box-shadow:0 10px 40px rgba(26,111,181,.3)}
#welcome h1{font-size:28px;font-weight:700;color:#fff;letter-spacing:-.6px;background:linear-gradient(135deg,#fff 0%,#9fc8f5 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
#welcome p{color:#8899bb;max-width:480px;line-height:1.75;font-size:14px}
#welcome .features{color:#9fb8d4;font-size:13px;display:flex;flex-direction:column;gap:8px;margin-top:4px}
#welcome .features span{padding:2px 0;opacity:.85}
#start-btn{margin-top:14px;padding:15px 44px;background:linear-gradient(135deg,#1a6fb5,#1e90ff);color:#fff;border:none;border-radius:12px;font-weight:700;font-size:15px;cursor:pointer;transition:all .25s;box-shadow:0 8px 24px rgba(26,111,181,.4);letter-spacing:.3px}
#start-btn:hover{background:linear-gradient(135deg,#1e90ff,#5bb8f5);transform:translateY(-2px);box-shadow:0 12px 30px rgba(91,184,245,.5)}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#1a2540;border-radius:3px}
textarea::placeholder{color:#445}
</style>
</head>
<body>
<div id="layout">

<aside id="side-panel" aria-label="Audit summary dashboard">
  <div class="sp-brandbar">
    <img src="/static/logo.jpg" alt="" onerror="this.style.display='none'">
    <span class="sp-brand-text">Live Audit Dashboard</span>
  </div>
  <div class="sp-client-card">
    <div class="sp-company" id="sp-company">&mdash;</div>
    <div class="sp-contact" id="sp-contact">&mdash;</div>
    <div class="sp-badges">
      <span class="sp-pill sp-pill-type" id="sp-pill-type">Classifying</span>
      <span class="sp-pill sp-pill-fit-unknown" id="sp-pill-fit">Assessing</span>
    </div>
  </div>
  <div class="sp-content">
    <div class="sp-section">
      <div class="sp-section-title"><span>Progress</span></div>
      <div class="sp-progress-wrap">
        <div class="sp-ring">
          <svg viewBox="0 0 70 70">
            <defs>
              <linearGradient id="spGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#5bb8f5"/>
                <stop offset="100%" stop-color="#1a6fb5"/>
              </linearGradient>
            </defs>
            <circle class="sp-ring-bg" cx="35" cy="35" r="30"/>
            <circle class="sp-ring-fg" id="sp-ring-fg" cx="35" cy="35" r="30" stroke-dasharray="188.5" stroke-dashoffset="188.5"/>
          </svg>
          <div class="sp-ring-text" id="sp-ring-text">0%</div>
        </div>
        <div class="sp-progress-meta">
          <div class="sp-current-stage" id="sp-current-stage">Getting Started</div>
          <div class="sp-stage-sub" id="sp-stage-sub">0 of 6 stages complete</div>
        </div>
      </div>
    </div>

    <div class="sp-section">
      <div class="sp-section-title"><span>Audit Pipeline</span></div>
      <div class="sp-pipeline" id="sp-pipeline"></div>
    </div>

    <div class="sp-section">
      <div class="sp-section-title"><span>Session Stats</span></div>
      <div class="sp-stats">
        <div class="sp-stat"><div class="sp-stat-value" id="sp-stat-questions">0</div><div class="sp-stat-label">Questions Answered</div></div>
        <div class="sp-stat"><div class="sp-stat-value" id="sp-stat-duration">0m</div><div class="sp-stat-label">Session Time</div></div>
        <div class="sp-stat"><div class="sp-stat-value" id="sp-stat-bottlenecks">0</div><div class="sp-stat-label">Bottlenecks</div></div>
        <div class="sp-stat"><div class="sp-stat-value" id="sp-stat-opportunities">0</div><div class="sp-stat-label">Opportunities</div></div>
      </div>
    </div>

    <div class="sp-section">
      <div class="sp-section-title"><span>Top Bottlenecks</span><span class="sp-count" id="sp-count-bottlenecks">0</span></div>
      <ul class="sp-list" id="sp-bottlenecks"><li class="sp-empty">Surfacing as you answer&hellip;</li></ul>
    </div>

    <div class="sp-section">
      <div class="sp-section-title"><span>Automation Opportunities</span><span class="sp-count" id="sp-count-opportunities">0</span></div>
      <ul class="sp-list" id="sp-opportunities"><li class="sp-empty">Matching workflows to your answers&hellip;</li></ul>
    </div>
  </div>
  <div class="sp-footer">
    <div class="sp-brand">Strat AI Solutions</div>
    <div class="sp-tagline">Operational intelligence for CRE finance</div>
    <a class="sp-link" id="sp-website-link" href="#" target="_blank" rel="noopener">
      <span>Visit our contact page</span>
      <span class="sp-link-arrow">&rarr;</span>
    </a>
  </div>
</aside>

<div id="app" style="position:relative">
  <div id="header">
    <div class="logo">
      <img src="/static/logo.jpg" alt="" aria-label="Strat AI" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
      <div class="logo-fallback" style="display:none" aria-hidden="true"></div>
    </div>
    <div id="header-right">
      <button id="export-btn" onclick="exportReport()">&#8595; Export Report</button>
      <div id="badge">Welcome</div>
    </div>
  </div>
  <div id="progress-bar"><div id="progress-fill"></div></div>
  <div id="messages">
    <div id="welcome">
      <img src="/static/logo.jpg" alt="Strat AI" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
      <div class="logo-fallback-big" style="display:none" aria-hidden="true"></div>
      <h1>Audit &amp; Scoping Bot</h1>
      <p>I'll identify your highest-value automation opportunities through a structured operational audit. We'll classify your operation, run a snapshot audit, synthesize bottlenecks, and build a clear scope.</p>
      <div class="features">
        <span>&#10003; Broker & Lender operations</span>
        <span>&#10003; Adaptive questioning -- 5-8 per batch</span>
        <span>&#10003; Bottleneck identification & ROI ranking</span>
        <span>&#10003; Matched to specific Sprint offerings</span>
        <span>&#10003; 30/60/90 implementation plan</span>
        <span>&#10003; Exportable proposal-ready report</span>
      </div>
      <button id="start-btn" onclick="showIntakeForm()">Start Audit &#8594;</button>
    </div>
  </div>
  <div id="input-area" style="display:none">
    <textarea id="msg-input" rows="3" placeholder="Type your answer here... (Shift+Enter for new line)"></textarea>
    <button id="send-btn" onclick="send()">Send</button>
  </div>
  <div id="info-bar" style="display:none">
    <span id="info-type">Type: &#8212;</span>
    <span id="info-snapshot">Snapshot: 0</span>
    <span id="info-flags">Flags: none</span>
    <span id="info-progress">Progress: 0%</span>
  </div>

  <!-- Item 17: Email capture overlay -->
  <div id="intake-overlay" style="display:none">
    <div id="intake-form">
      <img src="/static/logo.jpg" alt="Strat AI" style="width:48px;height:48px;border-radius:12px;margin-bottom:12px;object-fit:contain" onerror="this.style.display='none'">
      <h2>Before we begin</h2>
      <p>Enter your details so we can tailor the audit to your operation and send you the report when we're done.</p>
      <div class="intake-error" id="intake-error">Please fill in all fields.</div>
      <input type="text" id="intake-name" placeholder="Your full name" autocomplete="name">
      <input type="email" id="intake-email" placeholder="Work email" autocomplete="email">
      <input type="text" id="intake-company" placeholder="Company name" autocomplete="organization">
      <button id="intake-submit" onclick="submitIntake()">Continue to Audit &#8594;</button>
    </div>
  </div>
</div>

</div>
<script>
function showSidePanel(){document.getElementById('side-panel').classList.add('visible');}

function setPill(el,fit){
  el.className='sp-pill sp-pill-fit-'+(fit||'unknown');
}

function renderPipeline(stages){
  const el=document.getElementById('sp-pipeline');
  el.innerHTML='';
  (stages||[]).forEach(s=>{
    const row=document.createElement('div');
    row.className='sp-stage-row '+s.status;
    row.innerHTML=`<span class="sp-stage-dot"></span><span class="sp-stage-name">${s.label}</span><span class="sp-stage-check">&#10003;</span>`;
    el.appendChild(row);
  });
}

function updateSummaryPanel(summary){
  if(!summary)return;

  document.getElementById('sp-company').textContent=summary.company||'Your Company';
  document.getElementById('sp-contact').textContent=summary.contact_name?(summary.contact_name+(summary.contact_email?' \u00B7 '+summary.contact_email:'')):'';

  const tPill=document.getElementById('sp-pill-type');
  tPill.textContent=summary.client_type_label||'Classifying';

  const fPill=document.getElementById('sp-pill-fit');
  fPill.textContent=summary.fit_label||'Assessing';
  setPill(fPill,summary.fit);

  // Progress ring
  const pct=summary.progress!==undefined?summary.progress:0;
  const circumference=2*Math.PI*30;
  const offset=circumference-(pct/100)*circumference;
  const ring=document.getElementById('sp-ring-fg');
  ring.setAttribute('stroke-dasharray',circumference.toFixed(1));
  ring.setAttribute('stroke-dashoffset',offset.toFixed(1));
  document.getElementById('sp-ring-text').textContent=pct+'%';
  document.getElementById('sp-current-stage').textContent=summary.stage_label||'Getting Started';
  document.getElementById('sp-stage-sub').textContent=(summary.stages_completed||0)+' of '+(summary.stages_total||6)+' stages complete';

  renderPipeline(summary.stages);

  document.getElementById('sp-stat-questions').textContent=summary.questions_answered||0;
  document.getElementById('sp-stat-duration').textContent=(summary.duration_min||0)+'m';
  document.getElementById('sp-stat-bottlenecks').textContent=summary.bottleneck_count||0;
  document.getElementById('sp-stat-opportunities').textContent=summary.opportunity_count||0;

  document.getElementById('sp-count-bottlenecks').textContent=summary.bottleneck_count||0;
  document.getElementById('sp-count-opportunities').textContent=summary.opportunity_count||0;

  const bEl=document.getElementById('sp-bottlenecks');
  bEl.innerHTML='';
  if(summary.bottlenecks&&summary.bottlenecks.length){
    summary.bottlenecks.forEach(b=>{
      const li=document.createElement('li');
      li.textContent=b;
      bEl.appendChild(li);
    });
  }else{
    bEl.innerHTML='<li class="sp-empty">Surfacing as you answer&hellip;</li>';
  }

  const oEl=document.getElementById('sp-opportunities');
  oEl.innerHTML='';
  if(summary.opportunities&&summary.opportunities.length){
    summary.opportunities.forEach(o=>{
      const li=document.createElement('li');
      const nm=document.createElement('span');
      nm.className='opp-name';nm.textContent=o.name;
      const ds=document.createElement('span');
      ds.className='opp-desc';ds.textContent=o.desc||'';
      li.appendChild(nm);li.appendChild(ds);
      oEl.appendChild(li);
    });
  }else{
    oEl.innerHTML='<li class="sp-empty">Matching workflows to your answers&hellip;</li>';
  }

  if(summary.contact&&summary.contact.website){
    document.getElementById('sp-website-link').href=summary.contact.website;
  }
}
</script>
<script>
const sid=crypto.randomUUID();
let ws,started=false,calendlyUrl='';
const msgs=document.getElementById('messages');
const inp=document.getElementById('msg-input');
let contactInfo={};

// Item 17: Show intake form
function showIntakeForm(){
  document.getElementById('intake-overlay').style.display='flex';
}

function submitIntake(){
  const name=document.getElementById('intake-name').value.trim();
  const email=document.getElementById('intake-email').value.trim();
  const company=document.getElementById('intake-company').value.trim();
  const errEl=document.getElementById('intake-error');

  if(!name||!email||!company){
    errEl.textContent='Please fill in all fields.';
    errEl.style.display='block';
    return;
  }
  // Basic email validation
  if(!email.includes('@')||!email.includes('.')){
    errEl.textContent='Please enter a valid email address.';
    errEl.style.display='block';
    return;
  }
  errEl.style.display='none';
  contactInfo={name,email,company};
  document.getElementById('intake-overlay').style.display='none';
  // Seed the dashboard immediately with the info we already know
  document.getElementById('sp-company').textContent=company;
  document.getElementById('sp-contact').textContent=name+(email?' \u00B7 '+email:'');
  showSidePanel();
  startAudit();
}

let reconnects=0;
function connect(){
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(`${proto}//${location.host}/ws/${sid}`);
  ws.onopen=()=>{
    reconnects=0;
    // Item 6: Send contact info with initial connection
    if(contactInfo.name){
      ws.send(JSON.stringify({type:'intake',name:contactInfo.name,email:contactInfo.email,company:contactInfo.company}));
    }
  };
  ws.onmessage=e=>{
    let d;
    try{
      d=JSON.parse(e.data);
    }catch(err){
      // Item 3: Proper error handling for bad JSON
      console.error('Bad JSON from server:',e.data,err);
      hideTyping();
      document.getElementById('send-btn').disabled=false;
      addMsg('bot','Something went wrong processing the response. Please try sending your message again.');
      return;
    }
    // Item 2: Always hide typing indicator on any message
    hideTyping();
    document.getElementById('send-btn').disabled=false;
    if(d.type==='bot_message'){
      addMsg('bot',d.content,d);
      if(d.stage_label)document.getElementById('badge').textContent=d.stage_label;
      // Item 10: Progress tied to stage transition
      if(d.progress!==undefined)document.getElementById('progress-fill').style.width=d.progress+'%';
      document.getElementById('info-type').textContent='Type: '+(d.client_type!=='unknown'?d.client_type.toUpperCase():'\u2014');
      document.getElementById('info-snapshot').textContent='Snapshot: '+d.snapshot_count;
      document.getElementById('info-flags').textContent='Flags: '+(d.flags&&d.flags.length?d.flags.join(', '):'none');
      document.getElementById('info-progress').textContent='Progress: '+d.progress+'%';
      // Item 19: Calendly
      if(d.calendly_url)calendlyUrl=d.calendly_url;
      if(d.show_calendly&&d.calendly_url)showCalendly(d.calendly_url);
      // Show export once past synthesis
      if(d.progress>=66)document.getElementById('export-btn').style.display='block';
      // Live 'Summary So Far' side panel
      if(d.summary)updateSummaryPanel(d.summary);
    }else if(d.type==='error'){
      // Item 3: Show errors gracefully
      addMsg('bot',d.content||'Something went wrong. Please try again.');
    }
  };
  ws.onclose=()=>{
    if(!started)return;
    // Item 2: Always hide typing on close
    hideTyping();
    reconnects++;
    if(reconnects<=10)setTimeout(connect,3000);
    else{addMsg('bot','Connection lost. Please refresh the page to continue.');}
  };
  ws.onerror=()=>{
    hideTyping();
    ws.close();
  };
}

// Parse bot text into [intro, ...mcqBlocks, trailing]. Each mcqBlock = {stem, opts}.
// Recognizes option lines of form "A) ...", "B) ...", "C) ...", "D) ..." or "1. ..." through "4. ...".
function parseMCQBlocks(text){
  const optRe=/^\s*([A-Da-d][\)\.]|[1-4]\.)\s+(.+\S)\s*$/;
  const lines=text.split('\n');
  const blocks=[];
  let intro=[],trailing=[],curStem='',curOpts=[],mode='intro',stemBuffer=[];
  for(let i=0;i<lines.length;i++){
    const ln=lines[i];
    const m=ln.match(optRe);
    if(m){
      if(mode==='intro'){
        // The last non-empty line of intro becomes the question stem
        while(stemBuffer.length&&!stemBuffer[stemBuffer.length-1].trim())stemBuffer.pop();
        curStem=stemBuffer.length?stemBuffer.pop():'';
        intro=stemBuffer.slice();
        stemBuffer=[];
      }
      mode='opts';
      curOpts.push(m[0].trim());
    }else if(mode==='opts'){
      if(ln.trim()===''){continue;}
      // End of current MCQ block
      if(curOpts.length>=2){blocks.push({stem:curStem,opts:curOpts});}
      curStem='';curOpts=[];
      // Look ahead: if this line is a stem for the next MCQ, buffer it
      stemBuffer=[ln];
      mode='stem';
    }else if(mode==='stem'){
      if(ln.trim()===''){continue;}
      const m2=ln.match(optRe);
      if(m2){
        while(stemBuffer.length&&!stemBuffer[stemBuffer.length-1].trim())stemBuffer.pop();
        curStem=stemBuffer.length?stemBuffer.pop():'';
        stemBuffer=[];
        curOpts.push(m2[0].trim());
        mode='opts';
      }else{
        stemBuffer.push(ln);
      }
    }else{
      stemBuffer.push(ln);
    }
  }
  if(curOpts.length>=2){blocks.push({stem:curStem,opts:curOpts});}
  else if(stemBuffer.length){trailing=stemBuffer;}
  return{intro:intro.join('\n').trim(),blocks,trailing:trailing.join('\n').trim()};
}

function renderMCQBlock(stem,opts,container){
  const wrap=document.createElement('div');
  wrap.className='mcq-block';
  if(stem){
    const q=document.createElement('div');
    q.className='mcq-question';
    q.textContent=stem;
    wrap.appendChild(q);
  }
  const oc=document.createElement('div');
  oc.className='mcq-options';
  opts.forEach(opt=>{
    const btn=document.createElement('button');
    btn.className='mcq-btn';
    btn.textContent=opt;
    btn.onclick=()=>{
      oc.querySelectorAll('.mcq-btn').forEach(b=>{b.classList.remove('selected');b.disabled=true;});
      btn.classList.add('selected');
      btn.disabled=false;
      inp.value=opt;
      setTimeout(()=>send(),300);
    };
    oc.appendChild(btn);
  });
  wrap.appendChild(oc);
  container.appendChild(wrap);
}

function addMsg(role,text,meta){
  const d=document.createElement('div');
  d.className='msg '+role;
  const lbl=document.createElement('div');
  lbl.className='msg-label';
  lbl.textContent=role==='bot'?'Strat AI':'You';
  d.appendChild(lbl);
  // Show question batch progress chip during snapshot audit
  if(role==='bot'&&meta&&meta.stage==='snapshot'){
    const batchNum=(meta.snapshot_batch||0)+1;
    const chip=document.createElement('div');
    chip.className='stage-chip';
    chip.innerHTML=`<span class="chip-dot"></span>Snapshot Audit &mdash; Batch ${batchNum}`;
    d.appendChild(chip);
  }

  if(role==='bot'){
    const p=parseMCQBlocks(text);
    if(p.blocks.length>=1){
      if(p.intro){
        const b=document.createElement('div');
        b.textContent=p.intro;
        d.appendChild(b);
      }
      p.blocks.forEach(blk=>renderMCQBlock(blk.stem,blk.opts,d));
      if(p.trailing){
        const b=document.createElement('div');
        b.style.marginTop='8px';
        b.textContent=p.trailing;
        d.appendChild(b);
      }
    }else{
      const b=document.createElement('div');
      b.textContent=text;
      d.appendChild(b);
    }
  }else{
    const b=document.createElement('div');
    b.textContent=text;
    d.appendChild(b);
  }

  msgs.appendChild(d);
  msgs.scrollTop=msgs.scrollHeight;
}

// Item 19: Calendly banner
function showCalendly(url){
  if(document.getElementById('calendly-banner'))return;
  const d=document.createElement('div');
  d.id='calendly-banner';
  d.className='calendly-banner';
  d.innerHTML=`<a href="${url}" target="_blank" rel="noopener" onclick="trackCalendly()">Book Your 30-Minute Strategy Call &#8594;</a><p>Speak directly with our team about your audit results</p>`;
  msgs.appendChild(d);
  msgs.scrollTop=msgs.scrollHeight;
}

function trackCalendly(){
  // Item 19: Track Calendly clicks
  if(ws&&ws.readyState===1){
    ws.send(JSON.stringify({type:'calendly_click'}));
  }
}

function showTyping(){
  if(document.getElementById('typing-ind'))return;
  const d=document.createElement('div');
  d.id='typing-ind';
  d.innerHTML='<div id="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div><span>Analyzing...</span></div>';
  msgs.appendChild(d);
  msgs.scrollTop=msgs.scrollHeight;
  // Item 2: Auto-hide typing after 60s safety net
  setTimeout(()=>{hideTyping();},60000);
}
function hideTyping(){const e=document.getElementById('typing-ind');if(e)e.remove();}

function send(){
  const t=inp.value.trim();
  if(!t||!ws||ws.readyState!==1)return;
  addMsg('user',t);
  ws.send(JSON.stringify({content:t}));
  inp.value='';showTyping();
  document.getElementById('send-btn').disabled=true;
}

// Item 5 + 16: Export structured report from server
async function exportReport(){
  const btn=document.getElementById('export-btn');
  btn.textContent='Generating...';
  btn.disabled=true;
  try{
    const resp=await fetch(`/api/session/${sid}/report`);
    if(!resp.ok)throw new Error('Report generation failed');
    const data=await resp.json();
    const report=data.report||'Report generation failed.';
    const contact=data.contact||{};
    const date=new Date().toLocaleDateString('en-US',{year:'numeric',month:'long',day:'numeric'});

    const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    let fmtd=esc(report.trim());
    // Convert section headers (locked 8-section order, optional leading "N. ")
    fmtd=fmtd.replace(/^\s*(?:\d\.\s*)?(CLIENT OVERVIEW|TOP 3 BOTTLENECKS|TOP 3-5 AUTOMATION OPPORTUNITIES|RECOMMENDED ENGAGEMENT|PHASE 1 SCOPE|PROPOSED PHASE 1 SCOPE|PHASE 2 OPPORTUNITIES|SUCCESS METRICS|NEXT STEPS|NEXT STEP)\s*$/gm,'</div><h2>$1</h2><div class="section">');
    // Make URLs in the NEXT STEPS section clickable hyperlinks
    fmtd=fmtd.replace(/(https?:\/\/[^\s<"]+)/g,'<a href="$1" target="_blank" rel="noopener" style="color:#1a6fb5;font-weight:600;text-decoration:underline">Schedule a Discovery Call &rarr;</a>');

    const html=`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Strat AI Solutions \u2014 Audit & Scoping Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:#fff;color:#1a1a1a;max-width:900px;margin:0 auto;padding:48px 40px}
.doc-header{display:flex;align-items:center;gap:18px;padding-bottom:28px;border-bottom:3px solid #1a6fb5;margin-bottom:40px}
.doc-logo{width:54px;height:54px;background:linear-gradient(135deg,#0a4a7a,#1e90ff);border-radius:13px;flex-shrink:0;background-image:url('${location.origin}/static/logo.jpg');background-size:contain;background-repeat:no-repeat;background-position:center}
.doc-title{font-size:26px;font-weight:700;color:#0a1e3d;letter-spacing:-.4px}
.doc-badge{display:inline-block;background:#0a2a4a;color:#5bb8f5;font-size:10px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;padding:3px 10px;border-radius:20px;margin-left:10px;vertical-align:middle}
.doc-sub{font-size:13px;color:#888;margin-top:5px}
.section{line-height:1.78;font-size:14px;color:#333;white-space:pre-wrap;margin-bottom:12px}
h2{font-size:17px;font-weight:700;color:#0a4a7a;margin:32px 0 10px;padding-bottom:7px;border-bottom:1px solid #e5e7eb}
h3{font-size:14px;font-weight:600;color:#1a6fb5;margin:18px 0 6px}
strong{color:#1a1a1a;font-weight:600}
.doc-footer{margin-top:56px;padding-top:20px;border-top:1px solid #e5e7eb;display:flex;justify-content:space-between;font-size:12px;color:#aaa}
@media print{body{padding:20px}@page{margin:.75in}}
</style>
</head>
<body>
  <div class="doc-header">
    <div class="doc-logo" aria-label="Strat AI"></div>
    <div>
      <div class="doc-title">Strat AI Solutions<span class="doc-badge">Confidential</span></div>
      <div class="doc-sub">Audit &amp; Scoping Report \u2014 ${date}${contact.company?' \u2014 '+esc(contact.company):''}</div>
    </div>
  </div>
  <div class="section">${fmtd}</div>
  <div class="doc-footer">
    <span>&copy; ${new Date().getFullYear()} Strat AI Solutions \u2014 Confidential. Not for distribution.</span>
    <span>Generated ${new Date().toLocaleString()}</span>
  </div>
</body>
</html>`;
    const blob=new Blob([html],{type:'text/html'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;a.download=`strat-ai-audit-report${contact.company?'-'+contact.company.replace(/\s+/g,'-').toLowerCase():''}.html`;a.click();
    URL.revokeObjectURL(url);
  }catch(err){
    console.error('Export error:',err);
    addMsg('bot','Report export encountered an error. Please try again.');
  }finally{
    btn.textContent='\u2193 Export Report';
    btn.disabled=false;
  }
}

// Item 6: Fix start audit — reliable initialization
function startAudit(){
  started=true;
  const welcome=document.getElementById('welcome');
  if(welcome)welcome.remove();
  document.getElementById('input-area').style.display='flex';
  document.getElementById('info-bar').style.display='flex';
  document.getElementById('export-btn').style.display='block';
  showTyping();
  // Connect WebSocket — the onopen handler sends the initial message
  connect();
}

document.getElementById('msg-input').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}
});
</script>
</body>
</html>"""


# ===================================================================
# ADMIN DASHBOARD (Item 20)
# ===================================================================

ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Strat AI — Admin Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:#060b18;color:#e0e0e0;padding:24px}
h1{font-size:22px;color:#fff;margin-bottom:6px}
.subtitle{color:#6688aa;font-size:13px;margin-bottom:24px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:28px}
.stat-card{background:#0f1830;border:1px solid #1a2a50;border-radius:12px;padding:16px;text-align:center}
.stat-card .num{font-size:28px;font-weight:700;color:#5bb8f5}
.stat-card .label{font-size:11px;color:#6688aa;text-transform:uppercase;letter-spacing:.5px;margin-top:4px}
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;background:#0f1830;border-radius:12px;overflow:hidden;border:1px solid #1a2a50;min-width:960px}
th{background:#0a1e3d;color:#5bb8f5;font-size:11px;text-transform:uppercase;letter-spacing:.5px;padding:12px;text-align:left;font-weight:600;white-space:nowrap}
td{padding:10px 12px;font-size:13px;border-top:1px solid #1a2540;color:#bbb;vertical-align:top}
tr:hover td{background:#0a1e3d}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.3px}
.badge-good{background:#052e16;color:#4ade80}
.badge-moderate{background:#422006;color:#fbbf24}
.badge-poor{background:#450a0a;color:#f87171}
.badge-unknown{background:#1a2540;color:#6688aa}
.audit-pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:700;background:rgba(91,184,245,.12);color:#5bb8f5;border:1px solid rgba(91,184,245,.25)}
.hot{animation:hotglow 1.5s ease-in-out infinite alternate}
@keyframes hotglow{from{box-shadow:0 0 4px #4ade80}to{box-shadow:0 0 12px #4ade80}}
#login{display:flex;flex-direction:column;align-items:center;justify-content:center;height:80vh;gap:12px}
#login input{padding:11px 14px;background:#0f1830;border:1px solid #1a2a50;border-radius:8px;color:#e0e0e0;font-size:14px;width:260px;outline:none;font-family:inherit}
#login button{padding:11px 32px;background:#1a6fb5;color:#fff;border:none;border-radius:8px;font-weight:600;cursor:pointer;font-size:14px}
.error{color:#f87171;font-size:13px}
#dashboard{display:none}
.toolbar{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.refresh-btn{padding:6px 14px;background:#1a6fb5;color:#fff;border:none;border-radius:6px;font-size:12px;cursor:pointer;font-family:inherit}
.last-updated{font-size:11px;color:#445a7a}
</style>
</head>
<body>
<div id="login">
  <h1>Strat AI Admin</h1>
  <input type="password" id="pw" placeholder="Admin password" onkeydown="if(event.key==='Enter')login()">
  <button onclick="login()">Login</button>
  <div class="error" id="login-error"></div>
</div>
<div id="dashboard">
  <h1>Strat AI Solutions &mdash; Admin Dashboard</h1>
  <div class="subtitle">Session overview and lead management</div>
  <div class="toolbar">
    <button class="refresh-btn" onclick="loadData()">Refresh</button>
    <span class="last-updated" id="last-updated"></span>
  </div>
  <div class="stats" id="stats"></div>
  <div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Login Time</th>
        <th>Company</th>
        <th>Contact</th>
        <th>Client Type</th>
        <th>Fit</th>
        <th>Current Stage</th>
        <th>Top Bottleneck</th>
        <th>API Cost</th>
        <th>Call Booked</th>
        <th>Audits</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  </div>
</div>
<script>
let token='';
function fmtDate(str){
  if(!str)return'\u2014';
  const d=new Date(str);
  if(isNaN(d.getTime()))return String(str);
  return d.toLocaleString(undefined,{year:'numeric',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
}
async function login(){
  const pw=document.getElementById('pw').value;
  try{
    const r=await fetch('/admin/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
    const d=await r.json();
    if(d.ok){
      token=d.token;
      document.getElementById('login').style.display='none';
      document.getElementById('dashboard').style.display='block';
      loadData();
    }else{
      document.getElementById('login-error').textContent='Wrong password';
    }
  }catch(e){document.getElementById('login-error').textContent='Connection error';}
}
async function loadData(){
  try{
    const r=await fetch('/admin/api/sessions',{headers:{'Authorization':'Bearer '+token}});
    const d=await r.json();
    const sessions=d.sessions||[];

    // Stats
    const total=sessions.length;
    const completed=sessions.filter(s=>s.stage==='complete'||s.stage==='proposal_ready').length;
    const goodFit=sessions.filter(s=>s.fit==='good').length;
    const callsBooked=sessions.filter(s=>s.calendly_clicked).length;
    const totalCost=sessions.reduce((a,s)=>a+(s.api_cost_usd||0),0);
    document.getElementById('stats').innerHTML=`
      <div class="stat-card"><div class="num">${total}</div><div class="label">Total Sessions</div></div>
      <div class="stat-card"><div class="num">${completed}</div><div class="label">Completed</div></div>
      <div class="stat-card"><div class="num">${goodFit}</div><div class="label">Good Fit</div></div>
      <div class="stat-card"><div class="num">${callsBooked}</div><div class="label">Calls Booked</div></div>
      <div class="stat-card"><div class="num">$${totalCost.toFixed(2)}</div><div class="label">Total API Cost</div></div>
    `;
    document.getElementById('last-updated').textContent='Updated '+new Date().toLocaleTimeString();

    // Table
    const tbody=document.getElementById('tbody');
    tbody.innerHTML='';
    sessions.sort((a,b)=>new Date(b.created_at||0)-new Date(a.created_at||0));
    const STAGE_LABELS={
      'intake':'Getting Started','classify':'Classification','qualify':'Qualification',
      'snapshot':'Snapshot Audit','synthesis':'Synthesis','deep_audit':'Deep Audit',
      'proposal_ready':'Proposal Ready','complete':'Complete'
    };
    const TYPE_LABELS={'broker':'Broker','lender':'Lender','hybrid':'Hybrid','unknown':'\u2014'};
    const FIT_LABELS={'good':'Good','moderate':'Moderate','poor':'Poor','unknown':'\u2014'};
    sessions.forEach(s=>{
      const fit=s.fit||'unknown';
      const isHot=fit==='good';
      const tr=document.createElement('tr');
      if(isHot)tr.className='hot';
      const audits=s.user_audit_count||1;
      tr.innerHTML=`
        <td style="white-space:nowrap">${fmtDate(s.created_at)}</td>
        <td>${esc(s.company_name||'\u2014')}</td>
        <td>${esc(s.contact_name||'\u2014')}<br><small style="color:#6688aa">${esc(s.contact_email||'')}</small></td>
        <td>${TYPE_LABELS[s.client_type||'unknown']||'\u2014'}</td>
        <td><span class="badge badge-${fit}">${FIT_LABELS[fit]||'\u2014'}</span></td>
        <td>${STAGE_LABELS[s.stage]||esc(s.stage||'\u2014')}</td>
        <td style="max-width:180px;word-break:break-word">${s.flags&&s.flags.length?esc(s.flags[0]):'\u2014'}</td>
        <td>$${(s.api_cost_usd||0).toFixed(2)}</td>
        <td>${s.calendly_clicked?'<span style="color:#4ade80;font-weight:600">Yes</span>':'No'}</td>
        <td><span class="audit-pill">${audits}</span></td>
      `;
      tbody.appendChild(tr);
    });
  }catch(e){console.error(e);}
}
function esc(s){if(!s)return'';return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
</script>
</body>
</html>"""


# ===================================================================
# ROUTES
# ===================================================================

@app.get("/")
async def root():
    return HTMLResponse(FRONTEND_HTML)


@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = store.get_or_create(session_id)
    log.info(f"WS connected: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            # Item 3: Robust JSON parsing with fallback
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as e:
                log.warning(f"Bad JSON from client: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": "I had trouble understanding that message. Please try again.",
                })
                continue

            msg_type = msg.get("type", "chat")

            # Item 17: Handle intake info
            if msg_type == "intake":
                session.contact_name = msg.get("name", "")
                session.contact_email = msg.get("email", "")
                session.company_name = msg.get("company", "")
                session.stage = Stage.CLASSIFY
                log.info(f"Intake: {session.contact_name} / {session.contact_email} / {session.company_name}")
                # Send initial classification message
                result = await handle_message(
                    session,
                    f"I'm ready to start the audit. My name is {session.contact_name}, "
                    f"I'm from {session.company_name}. Please begin with the classification questions.",
                )
                await websocket.send_json(result)
                continue

            # Item 19: Track Calendly clicks
            if msg_type == "calendly_click":
                session.calendly_clicked = True
                log.info(f"Calendly clicked: {session.contact_email}")
                continue

            user_text = msg.get("content", "").strip()
            if not user_text:
                continue

            # Item 2: Send typing indicator acknowledgment
            result = await handle_message(session, user_text)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        log.info(f"WS disconnected: {session_id}")
    except Exception as e:
        log.error(f"WS error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": "Something went wrong. Please try sending your message again.",
            })
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/api/chat/{session_id}")
async def rest_chat(session_id: str, request: Request):
    body = await request.json()
    user_text = body.get("content", "").strip()
    if not user_text:
        raise HTTPException(400, "Empty message")
    session = store.get_or_create(session_id)
    result = await handle_message(session, user_text)
    return JSONResponse(result)


@app.post("/api/session/new")
async def new_session():
    sid = str(uuid.uuid4())
    store.get_or_create(sid)
    return {"session_id": sid}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.to_dict()


# Item 5 + 16: Server-side report generation from session data
@app.get("/api/session/{session_id}/report")
async def get_session_report(session_id: str):
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    report = await generate_structured_report(session)
    return {
        "report": report,
        "contact": {
            "name": session.contact_name,
            "email": session.contact_email,
            "company": session.company_name,
        },
        "session_id": session_id,
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.get("/api/session/{session_id}/export")
async def export_session(session_id: str):
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "export_date": datetime.utcnow().isoformat(),
        "session": session.to_dict(),
        "proposal_data": {
            "contact_name": session.contact_name,
            "contact_email": session.contact_email,
            "company_name": session.company_name,
            "client_type": session.client_type.value,
            "fit_status": session.fit.value,
            "qualification": session.qual_data,
            "snapshot_answers": session.snapshot_answers,
            "synthesis": session.synthesis_text[:5000] if session.synthesis_text else None,
            "deep_audit": session.deep_answers,
            "flags": session.flags,
            "recommended_workflows": _match_workflows(session),
            "api_cost_usd": round(session.api_cost_usd, 4),
        },
    }


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": store.all_sessions()}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "rag_ready": rag.ready,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "active_sessions": len(store._sessions),
        "has_anthropic_key": bool(ANTHROPIC_API_KEY),
    }


@app.post("/api/ingest")
async def ingest_document(request: Request):
    body = await request.json()
    coll = body.get("collection", "general")
    text = body.get("text", "")
    if not text:
        raise HTTPException(400, "No text provided")
    rag.ingest_text(coll, text)
    return {"status": "ok", "collection": coll}


# ===================================================================
# ADMIN ROUTES (Item 20)
# ===================================================================

# Simple token-based auth for admin
_admin_tokens: Dict[str, float] = {}


@app.get("/admin")
async def admin_page():
    return HTMLResponse(ADMIN_HTML)


@app.post("/admin/api/login")
async def admin_login(request: Request):
    body = await request.json()
    pw = body.get("password", "")
    if pw == ADMIN_PASSWORD:
        token = uuid.uuid4().hex
        _admin_tokens[token] = time.time()
        return {"ok": True, "token": token}
    return {"ok": False}


def _check_admin(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Unauthorized")
    token = auth[7:]
    if token not in _admin_tokens:
        raise HTTPException(401, "Invalid token")
    # Tokens expire after 24h
    if time.time() - _admin_tokens[token] > 86400:
        del _admin_tokens[token]
        raise HTTPException(401, "Token expired")
    return True


@app.get("/admin/api/sessions")
async def admin_sessions(request: Request):
    _check_admin(request)
    # Merge in-memory sessions with Supabase (Supabase wins on id conflict)
    mem_sessions = {s["id"]: s for s in store.all_sessions()}
    sb_sessions = {s["id"]: s for s in await load_supabase_sessions()}
    merged = {**mem_sessions, **sb_sessions}   # Supabase overwrites in-memory for same id
    sessions = list(merged.values())

    # Compute per-email audit counts for admin "Audit Count" column
    email_counts: Dict[str, int] = {}
    for s in sessions:
        e = (s.get("contact_email") or "").lower().strip()
        if e:
            email_counts[e] = email_counts.get(e, 0) + 1
    for s in sessions:
        e = (s.get("contact_email") or "").lower().strip()
        s["user_audit_count"] = email_counts.get(e, 1)

    return {"sessions": sessions}


@app.get("/admin/api/session/{session_id}")
async def admin_session_detail(session_id: str, request: Request):
    _check_admin(request)
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.to_dict()


# ===================================================================
# SSE STREAMING ENDPOINT (Item 24)
# ===================================================================

@app.get("/api/stream/{session_id}")
async def stream_chat(session_id: str, request: Request):
    """Item 24: SSE endpoint for streaming responses."""
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    async def event_generator():
        # Get the last user message
        user_msgs = [m for m in session.messages if m.get("role") == "user"]
        if not user_msgs:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        rag_context = ""
        if rag.ready:
            colls_map = {
                Stage.CLASSIFY: ["broker_audit", "lender_audit"],
                Stage.QUALIFY: ["sop", "strategy"],
                Stage.SNAPSHOT: ["broker_audit" if session.client_type != ClientType.LENDER else "lender_audit"],
                Stage.SYNTHESIS: ["workflows", "strategy"],
                Stage.DEEP_AUDIT: ["broker_audit" if session.client_type != ClientType.LENDER else "lender_audit"],
                Stage.PROPOSAL: ["workflows", "sop", "strategy"],
            }
            target = colls_map.get(session.stage, ["strategy"])
            last_msg = user_msgs[-1]["content"]
            rag_context = rag.query(last_msg, collections=target, n=4)

        system_prompt = build_full_system_prompt(session, rag_context)
        chat_msgs = []
        for m in session.messages[-MAX_HISTORY:]:
            role = "assistant" if m.get("role") == "bot" else "user"
            chat_msgs.append({"role": role, "content": m["content"]})

        deduped = []
        for m in chat_msgs:
            if deduped and deduped[-1]["role"] == m["role"]:
                deduped[-1]["content"] += "\n\n" + m["content"]
            else:
                deduped.append(m)

        if not deduped or deduped[0]["role"] != "user":
            deduped.insert(0, {"role": "user", "content": "Begin."})

        try:
            async with httpx.AsyncClient(timeout=90) as client:
                payload = {
                    "model": LLM_MODEL,
                    "max_tokens": 4096,
                    "system": system_prompt,
                    "messages": deduped,
                    "stream": True,
                }
                async with client.stream(
                    "POST",
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    full_text = ""
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                if chunk.get("type") == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    text = delta.get("text", "")
                                    if text:
                                        full_text += text
                                        clean = strip_markdown(text)
                                        yield f"data: {json.dumps({'type': 'token', 'text': clean})}\n\n"
                            except json.JSONDecodeError:
                                continue

                    # Process transitions
                    detect_transitions(full_text, session)
                    display_text = clean_control_tokens(full_text)
                    display_text = strip_markdown(display_text)

                    session.messages.append({
                        "role": "bot",
                        "content": display_text,
                        "ts": datetime.utcnow().isoformat(),
                    })

                    yield f"data: {json.dumps({'type': 'done', 'stage': session.stage.value, 'progress': session.progress_pct()})}\n\n"

        except Exception as e:
            log.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'text': 'Connection error. Please try again.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ===================================================================
# EMBEDDABLE WIDGET
# ===================================================================

WIDGET_JS_TEMPLATE = r"""
(function(){
  if(document.getElementById('strat-ai-widget'))return;
  var BOT_URL=window.STRAT_AI_BOT_URL||'__ORIGIN__';
  var open=false;
  var style=document.createElement('style');
  style.textContent='#strat-ai-btn{position:fixed;bottom:28px;right:28px;display:flex;align-items:center;gap:10px;padding:13px 20px;background:linear-gradient(135deg,#0a2a5a,#1a6fb5);color:#fff;border:none;border-radius:30px;cursor:pointer;font-family:-apple-system,BlinkMacSystemFont,Inter,sans-serif;font-size:14px;font-weight:700;letter-spacing:.2px;z-index:99999;box-shadow:0 6px 24px rgba(26,111,181,.55);transition:all .2s}#strat-ai-btn:hover{transform:translateY(-2px);box-shadow:0 10px 32px rgba(26,111,181,.7)}#strat-ai-frame{position:fixed;bottom:100px;right:28px;width:440px;height:680px;border:none;border-radius:16px;z-index:99998;box-shadow:0 12px 48px rgba(0,0,0,.55);display:none}#strat-ai-frame.sai-open{display:block;animation:sai-pop .25s cubic-bezier(.2,.7,.2,1)}@keyframes sai-pop{from{opacity:0;transform:translateY(12px) scale(.97)}to{opacity:1;transform:none}}@media(max-width:520px){#strat-ai-frame{width:calc(100vw - 16px);right:8px;bottom:90px;height:calc(100dvh - 110px)}}';
  document.head.appendChild(style);
  var ICON_CHAT='<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
  var ICON_CLOSE='<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  var btn=document.createElement('button');
  btn.id='strat-ai-btn';
  btn.setAttribute('aria-label','Start Strat AI Audit');
  btn.innerHTML=ICON_CHAT+'<span>Start Audit</span>';
  var frame=document.createElement('iframe');
  frame.id='strat-ai-frame';
  frame.title='Strat AI Audit & Scoping Bot';
  frame.src='about:blank';
  btn.addEventListener('click',function(){
    open=!open;
    if(open&&frame.src==='about:blank')frame.src=BOT_URL;
    frame.classList.toggle('sai-open',open);
    btn.innerHTML=open?(ICON_CLOSE+'<span>Close</span>'):(ICON_CHAT+'<span>Start Audit</span>');
  });
  var wrap=document.createElement('div');
  wrap.id='strat-ai-widget';
  wrap.appendChild(frame);
  wrap.appendChild(btn);
  document.body.appendChild(wrap);
})();
"""


@app.get("/widget.js")
async def widget_script(request: Request):
    origin = str(request.base_url).rstrip("/")
    js = WIDGET_JS_TEMPLATE.replace("__ORIGIN__", origin)
    return HTMLResponse(content=js, media_type="application/javascript")


# ===================================================================
# RUN
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting Strat AI Bot on {HOST}:{PORT}")
    log.info(f"LLM: {LLM_PROVIDER} / {LLM_MODEL}")
    log.info(f"RAG: {'enabled (local embeddings)' if HAS_CHROMA else 'disabled'}")
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        log.error("=" * 60)
        log.error("  ANTHROPIC_API_KEY is missing from .env!")
        log.error("  The bot will not work without it.")
        log.error("  Get your key at: https://console.anthropic.com")
        log.error("=" * 60)
    elif LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY.startswith("sk-ant-"):
        log.warning("ANTHROPIC_API_KEY doesn't start with 'sk-ant-' -- double check it's correct")
    log.info(f"Admin dashboard: http://localhost:{PORT}/admin")
    log.info(f"Open http://localhost:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
