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

# ChromaDB for RAG — disabled if DISABLE_RAG=true (saves ~400MB RAM on low-memory hosts)
_DISABLE_RAG = os.getenv("DISABLE_RAG", "").lower() in ("1", "true", "yes")
try:
    if _DISABLE_RAG:
        raise ImportError("RAG disabled via DISABLE_RAG env var")
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
BOT_PASSWORD = os.getenv("BOT_PASSWORD", "")
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
    feedback: Dict[str, Any] = field(default_factory=dict)

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

BASE_SYSTEM = """You are Strat AI Solutions' Audit Scoping Bot — an expert scoping assistant for CRE brokerages, mortgage brokerages, CRE lenders, and adjacent real estate operators. Built by Strat AI Solutions, founded by Yaseen Abdelrahman.

YOUR JOB: Classify → Qualify → Snapshot Audit → Synthesize → Deep Audit (targeted) → Proposal-Ready Output.

HARD RULES:
- Ask questions ONE AT A TIME. After the user answers, use their response to adapt and inform your next question. Never dump 6+ questions at once. If presenting 2-3 questions together, separate each with a full blank line and a divider line (─────────). Prefer multiple-choice format — always offer 4 labeled options (A, B, C, D) when applicable. Learn from each answer before moving on.
- Never skip classification or qualification.
- Never recommend automation without explaining the bottleneck first.
- Never hide uncertainty — state what is missing.
- Never call a poor fit a good fit.
- Never give generic AI ideas — tie every opportunity to a real workflow, user, system, and bottleneck.
- Push for specifics: volumes, cycle times, team roles, systems, error rates.
- Quantify impact: hours lost, delays, conversion loss, compliance risk.
- If the client is rambling, summarize and redirect.
- Follow bottleneck signals — pivot when something urgent surfaces.
- Short bullets over long essays.
- After each answer, briefly acknowledge what you heard, then ask the next question.
- Request artifacts when helpful: SOPs, templates, checklists, pipeline screenshots, email templates.
- Use markdown formatting for structure: **bold** for key terms and bottleneck names, *italic* for emphasis, ## for section headings, - for bullet lists, and numbered lists (1. 2. 3.) for sequences. Keep it scannable, not verbose.
- ALWAYS lead with the highest-revenue bottleneck first. The order of bottlenecks must reflect what makes Strat AI and the client the most money. Never bury the top opportunity.
- When the audit reaches proposal stage, ALWAYS direct the client to book their scoping call via the Calendly link. NEVER say "Yaseen will reach out to you" or "someone will contact you" — the client must click the link themselves to book.
- Do NOT present formal proposals, pricing, or engagement recommendations until the PROPOSAL stage. During synthesis and deep audit, focus only on identifying and quantifying bottlenecks.
- NEVER use -- (double hyphen) in your responses. Use — (em dash) instead.

TONE: Founder-friendly. Direct. Analytical. Commercially sharp. Not corporate. Not robotic.

OFFERINGS YOU CAN RECOMMEND:
1. Beta Jumpstart Sprint — Automate one bottleneck in one week, fixed price, guaranteed result
2. Command Center — Full workflow buildout across all identified bottlenecks
3. Custom Packages — Multi-scope bundles, bespoke automation builds
4. Strategic/Flagship Partnership — Co-building at scale, case study arrangements

BETA WORKFLOW SPRINTS (match to specific bottleneck):
1. Smart Document Collection — Portal + reminders, zero chasing
2. Executive Dashboards — Real-time pipeline & KPI visualization
3. Lead Follow-Up — 5-min first response, multi-channel (email/SMS/voicemail)
4. Onboarding Timeline — Forms, foldering, timeline tracking
5. Team Accountability — Daily outreach tracker & leaderboards
6. Scheduling & No-Show Recovery — Confirmations, reminders, auto-rebook
7. Pipeline/Deal Tracking — Automatic stage updates from rep and client actions

PATTERN RECOGNITION — Always watch for:
- No single source of truth
- Document chasing and incomplete submissions
- No proactive status alerts or visibility
- Task assignment and communication overhead
- Business logic in someone's head, not codified
- Reporting that depends on manual updates
- Founder/operator bottleneck — too much depends on one person

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

Ask ONE multiple-choice question at a time. Format each question as: one clear sentence on its own line, followed by exactly 4 options labeled "A) ...", "B) ...", "C) ...", "D) ..." — each option on its own line. No markdown, no bullets, no bold.

After the user answers, briefly acknowledge what they said (one sentence), then ask your next question — adapting it based on what you just learned. Keep questions focused and build on previous answers.

If you present 2 questions in the same message (only when necessary), separate them with a blank line and a divider line: ─────────

When you have enough information to classify, end your response with the exact text: CLASSIFICATION: BROKER or CLASSIFICATION: LENDER or CLASSIFICATION: HYBRID"""

    elif s.stage == Stage.QUALIFY:
        return f"""
CURRENT STAGE: QUALIFICATION GATE
Client classified as: {ct.upper()}

Ask ONE question at a time. After each answer, acknowledge briefly (one sentence), then ask the next — adapting based on what you've learned. Start with the most important factor given what you already know.

Qualification areas to cover (in natural order, not all at once):
- Decision-maker status (are they the person who signs off?)
- Other stakeholders involved
- Top 1-3 business challenges (specific, not vague)
- Target timeline for implementing a solution
- Budget status (approved, under discussion, not yet?)
- Company size + business model
- Prior AI/automation experience

Prefer multiple-choice questions (A/B/C/D) where possible. If you must ask 2 questions in one message, separate them with a blank line and a divider: ─────────

Flag POOR FIT if: no clear problem, no authority, no budget conversation, wants off-the-shelf SaaS, or unrealistic timeline. Be helpful but label the risk.

When you have enough to qualify, end with one of these exact tokens (based on your assessment):
QUALIFICATION: COMPLETE FIT: GOOD   (decision-maker, clear problem, budget discussion started, realistic timeline)
QUALIFICATION: COMPLETE FIT: MODERATE   (decision-maker but budget unclear, or problem vague but real)
QUALIFICATION: COMPLETE FIT: POOR   (not decision-maker, no clear problem, no budget, wants off-the-shelf SaaS)
QUALIFICATION: FLAG - [reason] FIT: POOR   (serious disqualifier found)"""

    elif s.stage == Stage.SNAPSHOT:
        answered = set(s.snapshot_answers.keys())
        batch = get_snapshot_batch(ct, s.snapshot_batch, answered)
        q_text = "\n".join(f"- {q['text']}" for q in batch) if batch else "No more questions."
        answered_count = len(s.snapshot_answers)

        sig_qs = []
        if s.snapshot_batch > 0:
            sig_qs = [
                "What would break first if your volume suddenly doubled?",
                "What does your team hate doing most — the task they'd automate tomorrow if they could?",
                "What would be catastrophic if it were automated incorrectly?",
            ]

        prev = json.dumps(s.snapshot_answers, indent=1)[:3000]

        sig_note = f"\nAlso consider weaving in one of these signature questions naturally when the moment is right:\n" + "\n".join(f"- {q}" for q in sig_qs) if sig_qs else ""

        return f"""
CURRENT STAGE: SNAPSHOT AUDIT — Batch {s.snapshot_batch + 1}
Client type: {ct.upper()}
Progress: {answered_count} answer batches collected so far.

Ask 1-2 questions from the list below per message. After the user responds, acknowledge what you heard (2-3 short bullets), then ask the next question — adapting your wording based on what you've learned. Never dump all questions at once.

Format each question as a clear sentence, followed by 4 options (A) B) C) D)) where applicable. If presenting 2 questions in one message, separate them with a blank line and a divider: ─────────

Questions to draw from (adapt naturally, don't read robotically):
{q_text}
{sig_note}

After each answer:
1. Summarize what you heard (short bullets)
2. Flag any bottleneck signals worth following
3. Note missing specifics (volumes, cycle times, team sizes, systems)

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
{ct.upper()} — explain why

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
One of: Sprint, Command Center, Discovery Audit, Accelerated Scoping — with rationale

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
CURRENT STAGE: DEEP AUDIT — Module: {current_mod} ({s.deep_module_idx + 1} of {len(mods)})
Client type: {ct.upper()}
Remaining modules: {mods[s.deep_module_idx:]}

Ask ONE question at a time for this module. After each answer, acknowledge what you heard (briefly), then ask the next — adapted to what you've just learned. Start with the most critical aspect of this module based on the snapshot audit so far.

Key areas to probe for this module:
{q_text}

Prefer multiple-choice (A/B/C/D) where applicable. For each answer, capture: current-state workflow, systems touched, handoffs, manual steps, failure points, time loss, duplicate entry, compliance risks, what must stay human, what could be automated.

If 2 questions must appear in the same message, separate with a blank line and a divider: ─────────

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
What exists today — systems, processes, team, volume, pain points

2. Desired Future State
What the operation should look like post-automation

3. Phase 1 Scope (30-day deliverable)
Highest-ROI build — specific features, integrations, and workflows
Map to specific Beta Workflow Sprint(s)

4. Phase 2 Opportunities (60-90 day expansion)
Next-priority builds after Phase 1 proves value

5. Assumptions & Dependencies
What must be true for this to work

6. Success Metrics
Specific KPIs: faster intake, shorter doc cycle, reduced manual follow-up, more deals per headcount, fewer missed tasks, better compliance, dashboard visibility

7. Recommended Offering
Sprint ($X range), Command Center ($X range), or Partnership — with rationale

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
                "max_tokens": 8192,
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
        return "The request took too long. Please try again — I'll work faster this time."
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return f"Connection error — please try again in a moment. ({type(e).__name__})"


async def stream_llm(system_prompt: str, messages: List[Dict[str, Any]],
                     session: Optional[Session] = None, websocket=None) -> str:
    """Stream LLM response using Anthropic's streaming API.
    Sends chunks to websocket in real-time if provided."""
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

    if not ANTHROPIC_API_KEY:
        return "ERROR: ANTHROPIC_API_KEY is not set. Please add it to your .env file."

    full_text = ""
    input_tokens = 0
    output_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            payload = {
                "model": LLM_MODEL,
                "max_tokens": 8192,
                "system": system_prompt,
                "messages": deduped[-MAX_HISTORY:],
                "stream": True,
            }
            log.info(f"Streaming Anthropic API -- model: {LLM_MODEL}, messages: {len(deduped)}")
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
                if resp.status_code != 200:
                    body = await resp.aread()
                    try:
                        data = json.loads(body)
                        err_msg = data.get("error", {}).get("message", "Unknown error")
                    except Exception:
                        err_msg = f"HTTP {resp.status_code}"
                    log.error(f"Anthropic streaming API error: {err_msg}")
                    return f"I encountered an issue connecting to the AI service. Please try again in a moment. (Error: {err_msg})"

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    chunk_type = chunk.get("type", "")

                    if chunk_type == "message_start":
                        usage = chunk.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

                    elif chunk_type == "message_delta":
                        usage = chunk.get("usage", {})
                        output_tokens = usage.get("output_tokens", 0)

                    elif chunk_type == "content_block_delta":
                        delta = chunk.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            full_text += text
                            if websocket is not None:
                                try:
                                    await websocket.send_json({"type": "chunk", "content": text})
                                except Exception:
                                    pass

        # Cost tracking
        if session:
            cost = (input_tokens * SONNET_INPUT_COST) + (output_tokens * SONNET_OUTPUT_COST)
            session.api_cost_usd += cost
            session.api_calls += 1
            log.info(f"Session {session.id[:8]}: +${cost:.4f} (total: ${session.api_cost_usd:.4f})")

        return full_text

    except httpx.TimeoutException:
        return "The request took too long. Please try again — I'll work faster this time."
    except Exception as e:
        log.error(f"Stream LLM call failed: {e}")
        return f"Connection error — please try again in a moment. ({type(e).__name__})"


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
This report will be read by a CRE principal — it must be clear, concise, and actionable.
They should be able to read it in under 5 minutes and know exactly what the problem is,
what the recommendation is, and what the next step is.

STRICT FORMATTING RULES (violating any of these will make the report unpresentable):
- NO markdown: no asterisks (*), no double-asterisks (**), no hashtags (#), no underscores (_), no dashes as bullets
- NO bullet points of any kind. Use numbered lists only.
- Section headers are plain text with no formatting symbols
- Plain text only throughout

Session data:
{json.dumps(all_data, indent=2)[:6000]}

Use this EXACT structure. Section order is LOCKED — do not reorder, rename, merge, or omit sections. Output sections 1-7 as normal. For section 8 (NEXT STEPS), output ONLY the literal static text provided below — do not generate dynamic content, do not reword, do not add context. Copy it exactly.

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
            # Detect fit assessment from qualify stage
            fit_match = re.search(r"FIT:\s*(GOOD|MODERATE|POOR)", text, re.IGNORECASE)
            if fit_match:
                fit_str = fit_match.group(1).lower()
                try:
                    session.fit = Fit(fit_str)
                except ValueError:
                    pass
            break
    return session


def clean_control_tokens(text: str) -> str:
    patterns = [
        r"CLASSIFICATION:\s*(BROKER|LENDER|HYBRID)\s*",
        r"QUALIFICATION:\s*(COMPLETE|FLAG[^\n]*)\s*",
        r"FIT:\s*(GOOD|MODERATE|POOR)\s*",
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
        """Remove in-memory sessions older than SESSION_TTL_HOURS."""
        cutoff = datetime.utcnow() - timedelta(hours=SESSION_TTL_HOURS)
        expired = []
        for sid, s in self._sessions.items():
            try:
                ts = (s.created_at or "").strip()
                if not ts:
                    continue
                # Normalise Z suffix for Python < 3.11 and strip timezone info
                # so we can compare against a naive UTC cutoff
                ts_norm = ts.replace("Z", "+00:00")[:26]
                dt = datetime.fromisoformat(ts_norm)
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)  # treat stored value as UTC, drop tz
                if dt < cutoff:
                    expired.append(sid)
            except Exception:
                pass  # never expire a session we can't parse — leave it in memory
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
        "feedback": session.feedback,
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

async def handle_message(session: Session, user_message: str, websocket=None) -> Dict[str, Any]:
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
    response = await stream_llm(system_prompt, session.messages[-MAX_HISTORY:], session=session, websocket=websocket)

    prev_stage = session.stage
    session = detect_transitions(response, session)
    display_text = clean_control_tokens(response)

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

    # Persist to Supabase after every message so no data is lost between restarts
    asyncio.create_task(save_to_supabase(session))

    summary = summarize_progress(session)
    if summary.get("bottlenecks"):
        session.metadata["top_bottleneck"] = summary["bottlenecks"][0]
    if summary.get("opportunities"):
        session.metadata["top_opportunity"] = summary["opportunities"][0]["name"]

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
        "summary": summary,
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
        "created_at": session.created_at or "",
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
    # Restore all persisted sessions from Supabase into the in-memory store
    if _supabase_client:
        try:
            prior = await load_supabase_sessions()
            loaded = 0
            for row in prior:
                try:
                    sid = row.get("id")
                    if not sid or sid in store._sessions:
                        continue
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
                    loaded += 1
                except Exception as row_err:
                    log.warning(f"Skipping malformed Supabase row {row.get('id', '?')}: {row_err}")
            log.info(f"Restored {loaded}/{len(prior)} sessions from Supabase")
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

/* MCQ question blocks */
.mcq-block{margin:16px 0;padding:16px 18px;background:rgba(10,16,38,.85);border:1px solid rgba(26,42,80,.9);border-left:3px solid rgba(91,184,245,.35);border-radius:12px}
.mcq-question{font-size:13.5px;font-weight:600;color:#d8e8ff;line-height:1.6;margin-bottom:12px}
.mcq-options{display:flex;flex-direction:column;gap:7px;margin-bottom:4px}
.mcq-opt{display:flex;align-items:center;gap:10px;padding:10px 14px;background:rgba(8,14,32,.7);border:1px solid rgba(26,42,80,.8);border-radius:8px;cursor:pointer;transition:all .15s;color:#8faac4;font-size:13px;user-select:none}
.mcq-opt:hover{border-color:rgba(91,184,245,.4);background:rgba(26,56,96,.4);color:#c8ddf0}
.mcq-opt.selected{border-color:#1a6fb5;background:rgba(26,111,181,.2);color:#d8e8ff}
.mcq-opt.locked{cursor:default;pointer-events:none}
.mcq-opt.locked:not(.selected){opacity:.45}
.mcq-opt-letter{width:26px;height:26px;border-radius:50%;background:rgba(26,64,110,.5);border:1px solid rgba(91,184,245,.25);color:#5bb8f5;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s}
.mcq-opt.selected .mcq-opt-letter{background:#1a6fb5;border-color:#5bb8f5;color:#fff}
.mcq-opt-text{flex:1;line-height:1.45}
.mcq-divider{border:none;border-top:1px solid rgba(26,42,80,.7);margin:14px 0}
.mcq-confirm{display:none;margin-top:12px;padding:9px 20px;background:linear-gradient(135deg,#1a6fb5,#1e90ff);color:#fff;border:none;border-radius:8px;font-weight:600;font-size:12.5px;cursor:pointer;transition:all .15s;letter-spacing:.3px;align-items:center;gap:6px}
.mcq-confirm:hover{background:linear-gradient(135deg,#1e90ff,#5bb8f5);transform:translateY(-1px)}
.mcq-confirm:disabled{opacity:.5;cursor:not-allowed;transform:none}

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
/* Streaming message */
.msg.streaming .stream-cursor{display:inline-block;width:2px;height:14px;background:#5bb8f5;margin-left:2px;animation:blink .7s step-end infinite;vertical-align:text-bottom}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
/* Markdown in bot messages */
.msg.bot strong{color:#7dd3fc;font-weight:700}
.msg.bot em{color:#c4b5fd;font-style:italic}
.msg.bot h2{font-size:14px;font-weight:700;color:#5bb8f5;margin:12px 0 6px;padding-bottom:4px;border-bottom:1px solid rgba(91,184,245,.2)}
.msg.bot h3{font-size:13px;font-weight:700;color:#93c5fd;margin:10px 0 4px}
.msg.bot ul,.msg.bot ol{margin:6px 0 6px 18px;display:flex;flex-direction:column;gap:3px}
.msg.bot li{font-size:13px;color:#d0d8e8;line-height:1.5}
.msg.bot p{margin:4px 0;color:#d0d8e8;line-height:1.65}
/* Report canvas drawer */
#report-canvas{position:fixed;right:-520px;top:0;width:500px;height:100vh;background:#0a1428;border-left:1px solid rgba(91,184,245,.25);z-index:300;transition:right .4s cubic-bezier(.2,.7,.2,1);display:flex;flex-direction:column;box-shadow:-8px 0 40px rgba(0,0,0,.5)}
#report-canvas.open{right:0}
#report-canvas-header{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid rgba(26,42,80,.6);flex-shrink:0}
#report-canvas-header h2{font-size:15px;font-weight:700;color:#fff}
#report-canvas-close{background:none;border:none;color:#6688aa;font-size:22px;cursor:pointer;line-height:1;padding:0 4px}
#report-canvas-close:hover{color:#fff}
#report-canvas-body{flex:1;overflow-y:auto;padding:20px}
#report-canvas-body h2{font-size:13px;font-weight:700;color:#5bb8f5;margin:16px 0 8px;padding-bottom:4px;border-bottom:1px solid rgba(26,42,80,.6)}
#report-canvas-body p{font-size:12.5px;color:#bcc8dc;line-height:1.7;white-space:pre-wrap;margin-bottom:10px}
#report-canvas-footer{padding:14px 20px;border-top:1px solid rgba(26,42,80,.6);display:flex;gap:10px;flex-shrink:0}
#report-canvas-export{flex:1;padding:11px;background:linear-gradient(135deg,#1a6fb5,#1e90ff);color:#fff;border:none;border-radius:10px;font-weight:700;font-size:13px;cursor:pointer;transition:all .2s}
#report-canvas-export:hover{background:linear-gradient(135deg,#1e90ff,#5bb8f5)}
/* Completion celebration */
.audit-complete-banner{margin:16px 0;padding:20px 22px;background:linear-gradient(135deg,rgba(74,222,128,.12),rgba(16,185,129,.08));border:1px solid rgba(74,222,128,.35);border-radius:14px;text-align:center}
.audit-complete-banner .congrats-title{font-size:17px;font-weight:700;color:#4ade80;margin-bottom:8px}
.audit-complete-banner .congrats-sub{font-size:13px;color:#9fb8d4;line-height:1.65;margin-bottom:14px}
.btn-export-report{display:inline-block;padding:12px 28px;background:linear-gradient(135deg,#1a6fb5,#1e90ff);color:#fff;border:none;border-radius:10px;font-weight:700;font-size:14px;cursor:pointer;transition:all .2s;box-shadow:0 6px 20px rgba(26,111,181,.4);animation:pulse-cta 2s ease-in-out infinite}
@keyframes pulse-cta{0%,100%{transform:scale(1);box-shadow:0 6px 20px rgba(26,111,181,.4)}50%{transform:scale(1.03);box-shadow:0 8px 28px rgba(91,184,245,.5)}}
.btn-export-report:hover{background:linear-gradient(135deg,#1e90ff,#5bb8f5)}
/* Feedback form */
#feedback-section{margin:16px 0;padding:20px;background:#0c1830;border:1px solid rgba(26,42,80,.8);border-radius:14px}
#feedback-section h3{font-size:13px;font-weight:700;color:#5bb8f5;margin-bottom:12px}
.stars{display:flex;gap:6px;margin-bottom:12px;font-size:26px;cursor:pointer}
.star{color:#1a2540;transition:color .15s;user-select:none}
.star.on{color:#fbbf24}
#feedback-comment{width:100%;background:#0a1428;border:1px solid rgba(26,42,80,.8);border-radius:8px;color:#e0e0e0;padding:10px 14px;font-size:13px;font-family:inherit;resize:none;outline:none;margin-bottom:10px}
#feedback-comment:focus{border-color:#5bb8f5}
#feedback-submit{padding:9px 22px;background:#1a6fb5;color:#fff;border:none;border-radius:8px;font-weight:600;font-size:13px;cursor:pointer;transition:all .2s}
#feedback-submit:hover{background:#1e90ff}
#feedback-thanks{color:#4ade80;font-size:13px;display:none;margin-top:8px}
#pw-gate{position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;background:radial-gradient(ellipse at top,#0a1428 0%,#060b18 60%)}
#pw-box{background:#0f1830;border:1px solid #1a2a50;border-radius:16px;padding:40px 36px;width:340px;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.5)}
#pw-box .pw-brand{font-size:18px;font-weight:700;color:#5bb8f5;margin-bottom:6px}
#pw-box .pw-sub{font-size:12px;color:#6688aa;margin-bottom:24px}
#pw-inp{width:100%;padding:11px 14px;background:#0a1428;border:1px solid #1a2a50;border-radius:8px;color:#e0e0e0;font-size:14px;font-family:inherit;outline:none;transition:border .15s;margin-bottom:12px}
#pw-inp:focus{border-color:#5bb8f5}
#pw-submit{width:100%;padding:11px;background:linear-gradient(135deg,#0a2a5a,#1a6fb5);color:#fff;border:none;border-radius:8px;font-weight:600;font-size:14px;cursor:pointer;font-family:inherit;transition:opacity .15s}
#pw-submit:hover{opacity:.85}
#pw-err{color:#f87171;font-size:12px;margin-top:8px;min-height:16px}
</style>
</head>
<body>
<div id="pw-gate" style="display:none">
  <div id="pw-box">
    <div class="pw-brand">Strat AI Solutions</div>
    <div class="pw-sub">Enter your access password to continue</div>
    <input type="password" id="pw-inp" placeholder="Password" onkeydown="if(event.key==='Enter')submitPw()">
    <button id="pw-submit" onclick="submitPw()">Access Bot</button>
    <div id="pw-err"></div>
  </div>
</div>
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
        <span>&#10003; Adaptive questioning &#8212; one question at a time</span>
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

  <div id="report-canvas">
    <div id="report-canvas-header">
      <h2>&#128196; Audit Report</h2>
      <button id="report-canvas-close" onclick="closeReportCanvas()">&#10005;</button>
    </div>
    <div id="report-canvas-body"></div>
    <div id="report-canvas-footer">
      <button id="report-canvas-export" onclick="exportReport()">&#8595; Download Report</button>
    </div>
  </div>
</div>

</div>
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<script>
// Configure marked
marked.setOptions({breaks: true, gfm: true});

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

// Real-time elapsed timer
let sessionStartTime = null;
function startElapsedTimer(){
  sessionStartTime = Date.now();
  setInterval(()=>{
    if(!sessionStartTime)return;
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 60000);
    const el = document.getElementById('sp-stat-duration');
    if(el) el.textContent = elapsed + 'm';
  }, 15000);
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

  // Stats — questions_answered and duration updated from server data
  document.getElementById('sp-stat-questions').textContent=summary.questions_answered||0;
  // Sync timer to actual session start time from server on first update
  if(!sessionStartTime&&summary.created_at){
    sessionStartTime=new Date(summary.created_at+'Z').getTime();
  }
  if(!sessionStartTime){
    document.getElementById('sp-stat-duration').textContent=(summary.duration_min||0)+'m';
  }
  document.getElementById('sp-stat-bottlenecks').textContent=summary.bottleneck_count||0;
  document.getElementById('sp-stat-opportunities').textContent=summary.opportunity_count||0;

  document.getElementById('sp-count-bottlenecks').textContent=summary.bottleneck_count||0;
  document.getElementById('sp-count-opportunities').textContent=summary.opportunity_count||0;

  const bEl=document.getElementById('sp-bottlenecks');
  bEl.innerHTML='';
  if(summary.bottlenecks&&summary.bottlenecks.length){
    summary.bottlenecks.forEach((b,i)=>{
      const li=document.createElement('li');
      li.textContent=b;
      if(i===0)li.style.borderLeft='2px solid #5bb8f5';
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

// Streaming state
let streamEl=null;
let streamText='';

function showIntakeForm(){
  document.getElementById('intake-overlay').style.display='flex';
}

function submitIntake(){
  const name=document.getElementById('intake-name').value.trim();
  const email=document.getElementById('intake-email').value.trim();
  const company=document.getElementById('intake-company').value.trim();
  const errEl=document.getElementById('intake-error');
  if(!name||!email||!company){
    errEl.textContent='Please fill in all fields.';errEl.style.display='block';return;
  }
  if(!email.includes('@')||!email.includes('.')){
    errEl.textContent='Please enter a valid email address.';errEl.style.display='block';return;
  }
  errEl.style.display='none';
  contactInfo={name,email,company};
  document.getElementById('intake-overlay').style.display='none';
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
    if(contactInfo.name){
      ws.send(JSON.stringify({type:'intake',name:contactInfo.name,email:contactInfo.email,company:contactInfo.company}));
    }
  };
  ws.onmessage=e=>{
    let d;
    try{d=JSON.parse(e.data);}catch(err){
      console.error('Bad JSON:',e.data,err);
      hideTyping();finalizeStream();
      document.getElementById('send-btn').disabled=false;
      addMsg('bot','Something went wrong. Please try again.');return;
    }

    if(d.type==='chunk'){
      // Streaming chunk \u2014 accumulate and show
      streamText+=d.content;
      if(!streamEl){
        hideTyping();
        streamEl=createStreamingEl();
      }
      updateStreamingEl(streamEl, streamText);
      msgs.scrollTop=msgs.scrollHeight;
      return;
    }

    // Final message or error
    hideTyping();
    finalizeStream();
    document.getElementById('send-btn').disabled=false;

    if(d.type==='bot_message'){
      addMsg('bot',d.content,d);
      if(d.stage_label)document.getElementById('badge').textContent=d.stage_label;
      if(d.progress!==undefined)document.getElementById('progress-fill').style.width=d.progress+'%';
      document.getElementById('info-type').textContent='Type: '+(d.client_type!=='unknown'?d.client_type.toUpperCase():'\u2014');
      document.getElementById('info-snapshot').textContent='Snapshot: '+d.snapshot_count;
      document.getElementById('info-flags').textContent='Flags: '+(d.flags&&d.flags.length?d.flags.join(', '):'none');
      document.getElementById('info-progress').textContent='Progress: '+d.progress+'%';
      if(d.calendly_url)calendlyUrl=d.calendly_url;
      if(d.show_calendly&&d.calendly_url)showCalendly(d.calendly_url);
      if(d.progress>=66)document.getElementById('export-btn').style.display='block';
      if(d.summary)updateSummaryPanel(d.summary);
      // Show completion celebration
      if(d.stage==='complete'||d.stage==='proposal_ready'){
        showAuditComplete();
      }
    }else if(d.type==='error'){
      addMsg('bot',d.content||'Something went wrong. Please try again.');
    }
  };
  ws.onclose=()=>{
    if(!started)return;
    hideTyping();finalizeStream();
    reconnects++;
    if(reconnects<=10)setTimeout(connect,3000);
    else{addMsg('bot','Connection lost. Please refresh the page.');}
  };
  ws.onerror=()=>{hideTyping();finalizeStream();ws.close();};
}

function createStreamingEl(){
  const d=document.createElement('div');
  d.className='msg bot streaming';
  const lbl=document.createElement('div');
  lbl.className='msg-label';lbl.textContent='Strat AI';
  d.appendChild(lbl);
  const body=document.createElement('div');
  body.className='stream-body';
  d.appendChild(body);
  const cursor=document.createElement('span');
  cursor.className='stream-cursor';
  d.appendChild(cursor);
  msgs.appendChild(d);
  msgs.scrollTop=msgs.scrollHeight;
  return d;
}

function updateStreamingEl(el, text){
  const body=el.querySelector('.stream-body');
  if(!body)return;
  // Render markdown progressively
  try{
    body.innerHTML=marked.parse(text,{breaks:true,gfm:true});
  }catch(e){
    body.textContent=text;
  }
}

function finalizeStream(){
  if(streamEl){streamEl.remove();streamEl=null;}
  streamText='';
}

// Only A-D are treated as MCQ options (NOT 1-4 numbered lists)
function parseMCQBlocks(text){
  const optRe=/^\s*([A-Da-d][\)\.:])\s+(.+\S)\s*$/;
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
  let selectedOpt=null;
  opts.forEach(opt=>{
    const item=document.createElement('div');
    item.className='mcq-opt';
    const raw=opt.trim();
    const m=raw.match(/^([A-Da-d])[).\s:]\s*(.+)$/);
    if(m){
      const lbl=document.createElement('span');
      lbl.className='mcq-opt-letter';
      lbl.textContent=m[1].toUpperCase();
      const txt=document.createElement('span');
      txt.className='mcq-opt-text';
      txt.textContent=m[2];
      item.appendChild(lbl);
      item.appendChild(txt);
    }else{
      const txt=document.createElement('span');
      txt.className='mcq-opt-text';
      txt.textContent=raw;
      item.appendChild(txt);
    }
    item.onclick=()=>{
      oc.querySelectorAll('.mcq-opt').forEach(o=>o.classList.remove('selected'));
      item.classList.add('selected');
      selectedOpt=raw;
      confirmBtn.style.display='flex';
    };
    oc.appendChild(item);
  });
  wrap.appendChild(oc);
  const confirmBtn=document.createElement('button');confirmBtn.className='mcq-confirm';
  confirmBtn.innerHTML='Submit answer &rarr;';
  confirmBtn.onclick=()=>{
    if(!selectedOpt)return;
    oc.querySelectorAll('.mcq-opt').forEach(o=>o.classList.add('locked'));
    confirmBtn.disabled=true;confirmBtn.style.display='none';sendDirect(selectedOpt);
  };
  wrap.appendChild(confirmBtn);container.appendChild(wrap);
}

function sendDirect(t){
  if(!t||!ws||ws.readyState!==1)return;
  addMsg('user',t);ws.send(JSON.stringify({content:t}));showTyping();
  document.getElementById('send-btn').disabled=true;
}

function renderMarkdownContent(text, container){
  try{
    // Auto-bold A/B/C concern labels like "A.", "B.", "C." at line start or "(A)", "(B)", "(C)"
    let t=text.replace(/^([A-C][.)]\s)/gm,'**$1**').replace(/\(([A-C])\)/g,'**($1)**');
    const html=marked.parse(t,{breaks:true,gfm:true});
    const div=document.createElement('div');div.innerHTML=html;container.appendChild(div);
  }catch(e){
    const div=document.createElement('div');div.textContent=text;container.appendChild(div);
  }
}

function addMsg(role,text,meta){
  const d=document.createElement('div');
  d.className='msg '+role;
  const lbl=document.createElement('div');lbl.className='msg-label';
  lbl.textContent=role==='bot'?'Strat AI':'You';
  d.appendChild(lbl);
  if(role==='bot'&&meta&&meta.stage==='snapshot'){
    const batchNum=(meta.snapshot_batch||0)+1;
    const chip=document.createElement('div');chip.className='stage-chip';
    chip.innerHTML=`<span class="chip-dot"></span>Snapshot Audit — Batch ${batchNum}`;
    d.appendChild(chip);
  }
  if(role==='bot'){
    const p=parseMCQBlocks(text);
    if(p.blocks.length>=1){
      if(p.intro){renderMarkdownContent(p.intro,d);}
      p.blocks.forEach((blk,i)=>{
        if(i>0){const hr=document.createElement('hr');hr.className='mcq-divider';d.appendChild(hr);}
        renderMCQBlock(blk.stem,blk.opts,d);
      });
      if(p.trailing){
        const b=document.createElement('div');b.style.marginTop='10px';
        renderMarkdownContent(p.trailing,b);d.appendChild(b);
      }
    }else{
      renderMarkdownContent(text,d);
    }
  }else{
    const b=document.createElement('div');b.textContent=text;d.appendChild(b);
  }
  msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
}

function showCalendly(url){
  if(document.getElementById('calendly-banner'))return;
  const d=document.createElement('div');d.id='calendly-banner';d.className='calendly-banner';
  d.innerHTML=`<a href="${url}" target="_blank" rel="noopener" onclick="trackCalendly()">&#128197; Book Your 30-Minute Scoping Call &rarr;</a><p>Click the link above to choose your time directly — no waiting for a callback</p>`;
  msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
}

function trackCalendly(){
  if(ws&&ws.readyState===1){ws.send(JSON.stringify({type:'calendly_click'}));}
}

let auditCompleteShown=false;
function showAuditComplete(){
  if(auditCompleteShown)return;
  auditCompleteShown=true;
  setTimeout(()=>{
    const d=document.createElement('div');d.className='audit-complete-banner';
    d.innerHTML=`<div class="congrats-title">&#127881; Audit Complete!</div><div class="congrats-sub">Your operational audit is done. We've identified your top bottlenecks and automation opportunities.<br><strong>Click the &#8595; Export Report button at the top to download your report</strong>, or use the button below to view it first.</div><button class="btn-export-report" onclick="openReportCanvas()">&#128196; View &amp; Export Your Report &rarr;</button>`;
    msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
    showFeedbackForm();
  }, 1200);
}

// Report canvas
let reportData=null;
async function openReportCanvas(){
  const canvas=document.getElementById('report-canvas');
  const body=document.getElementById('report-canvas-body');
  body.innerHTML='<p style="color:#6688aa">Loading your report...</p>';
  canvas.classList.add('open');
  if(!reportData){
    try{
      const resp=await fetch(`/api/session/${sid}/report`);
      if(!resp.ok)throw new Error('failed');
      reportData=await resp.json();
    }catch(err){
      body.innerHTML='<p style="color:#f87171">Failed to load report. Please try again.</p>';return;
    }
  }
  renderReportCanvas(reportData);
}

function renderReportCanvas(data){
  const body=document.getElementById('report-canvas-body');
  const report=data.report||'';
  let html=report.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  html=html.replace(/^\s*(?:\d\.\s*)?(CLIENT OVERVIEW|TOP 3 BOTTLENECKS|TOP 3-5 AUTOMATION OPPORTUNITIES|RECOMMENDED ENGAGEMENT|PHASE 1 SCOPE|PHASE 2 OPPORTUNITIES|SUCCESS METRICS|NEXT STEPS)\s*$/gm,'<h2>$1</h2>');
  html=html.replace(/(https?:\/\/[^\s<"]+)/g,'<a href="$1" target="_blank" rel="noopener" style="color:#5bb8f5;font-weight:600">Book Scoping Call &rarr;</a>');
  html=html.replace(/\n/g,'<br>');
  body.innerHTML=`<div style="font-size:12.5px;color:#c4d4e8;line-height:1.75">${html}</div>`;
}

function closeReportCanvas(){document.getElementById('report-canvas').classList.remove('open');}

function showFeedbackForm(){
  if(document.getElementById('feedback-section'))return;
  const sec=document.createElement('div');sec.id='feedback-section';
  sec.innerHTML=`<h3>How was your audit experience?</h3>
  <div class="stars" id="star-row">
    <span class="star" data-v="1">&#9733;</span>
    <span class="star" data-v="2">&#9733;</span>
    <span class="star" data-v="3">&#9733;</span>
    <span class="star" data-v="4">&#9733;</span>
    <span class="star" data-v="5">&#9733;</span>
  </div>
  <textarea id="feedback-comment" rows="3" placeholder="Any comments or suggestions? (optional)"></textarea>
  <button id="feedback-submit" onclick="submitFeedback()">Submit Feedback</button>
  <div id="feedback-thanks">Thank you for your feedback!</div>`;
  msgs.appendChild(sec);msgs.scrollTop=msgs.scrollHeight;
  let rating=0;
  const stars=sec.querySelectorAll('.star');
  stars.forEach(s=>{
    s.onmouseover=()=>stars.forEach(x=>{x.classList.toggle('on',parseInt(x.dataset.v)<=parseInt(s.dataset.v));});
    s.onmouseout=()=>stars.forEach(x=>{x.classList.toggle('on',parseInt(x.dataset.v)<=rating);});
    s.onclick=()=>{rating=parseInt(s.dataset.v);stars.forEach(x=>{x.classList.toggle('on',parseInt(x.dataset.v)<=rating);});};
  });
  sec.querySelector('#feedback-submit').addEventListener('click',async()=>{
    const comment=sec.querySelector('#feedback-comment').value.trim();
    if(rating===0)return;
    try{
      await fetch(`/api/feedback/${sid}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rating,comment})});
      sec.querySelector('#feedback-thanks').style.display='block';
      sec.querySelector('#feedback-submit').disabled=true;
    }catch(e){console.error(e);}
  });
}

function showTyping(){
  if(document.getElementById('typing-ind'))return;
  const d=document.createElement('div');d.id='typing-ind';
  d.innerHTML='<div id="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div><span>Generating response…</span></div>';
  msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
  // Extended timeout matching server timeout
  setTimeout(()=>{hideTyping();},120000);
}
function hideTyping(){const e=document.getElementById('typing-ind');if(e)e.remove();}

function send(){
  const t=inp.value.trim();
  if(!t||!ws||ws.readyState!==1)return;
  addMsg('user',t);ws.send(JSON.stringify({content:t}));inp.value='';showTyping();
  document.getElementById('send-btn').disabled=true;
}

// Export report — uses cached reportData if available
async function exportReport(){
  if(!reportData){
    const btn=document.getElementById('export-btn');
    const prevText=btn.textContent;
    btn.textContent='Fetching...';btn.disabled=true;
    try{
      const r=await fetch(`/api/session/${sid}/report`);
      if(!r.ok)throw new Error('failed');
      reportData=await r.json();
    }catch(e){addMsg('bot','Could not fetch report. Try again.');btn.textContent=prevText;btn.disabled=false;return;}
    finally{btn.disabled=false;}
  }
  const btn=document.getElementById('export-btn');
  const prevText=btn.textContent;
  btn.textContent='Building...';btn.disabled=true;
  try{
    const data=reportData;
    const report=data.report||'Report generation failed.';
    const contact=data.contact||{};
    const date=new Date().toLocaleDateString('en-US',{year:'numeric',month:'long',day:'numeric'});

    const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const md=typeof marked!=='undefined'?marked.parse(report):`<pre>${esc(report)}</pre>`;

    const coName=contact.company?'-'+contact.company.replace(/\s+/g,'-').toLowerCase():'';
    const html=`<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Strat AI Report${contact.company?' - '+esc(contact.company):''}</title>
<style>body{font-family:Inter,sans-serif;background:#fff;color:#1a1a1a;max-width:860px;margin:0 auto;padding:40px}h1{color:#0a4a7a;border-bottom:2px solid #1a6fb5;padding-bottom:8px}h2{color:#0a4a7a;margin-top:28px}p,li{line-height:1.7}footer{margin-top:48px;color:#aaa;font-size:12px;border-top:1px solid #eee;padding-top:12px}@media print{body{padding:20px}}</style>
</head><body>
<h1>Strat AI Solutions &mdash; Audit Report</h1>
<p style="color:#888;font-size:13px">${date}${contact.company?' &mdash; '+esc(contact.company):''}</p>
<div id="body">${md}</div>
<footer>&copy; ${new Date().getFullYear()} Strat AI Solutions &mdash; Confidential</footer>
</body></html>`;
    const blob=new Blob([html],{type:'text/html'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;a.download=`strat-ai-report${coName}.html`;a.click();
    URL.revokeObjectURL(url);
  }catch(err){
    console.error('Export error:',err);
    addMsg('bot','Report export encountered an error. Please try again.');
  }finally{
    btn.textContent=prevText;
    btn.disabled=false;
  }
}

function startAudit(){
  started=true;
  const welcome=document.getElementById('welcome');
  if(welcome)welcome.remove();
  document.getElementById('input-area').style.display='flex';
  document.getElementById('info-bar').style.display='flex';
  document.getElementById('export-btn').style.display='block';
  startElapsedTimer();
  showTyping();connect();
}

document.getElementById('msg-input').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}
});

// Password gate
async function checkBotPassword(){
  const saved=sessionStorage.getItem('bot_pw')||'';
  try{
    const r=await fetch('/api/verify-password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:saved})});
    const d=await r.json();
    if(d.ok){document.getElementById('pw-gate').style.display='none';return;}
  }catch(e){}
  document.getElementById('pw-gate').style.display='flex';
}
async function submitPw(){
  const pw=document.getElementById('pw-inp').value;
  const errEl=document.getElementById('pw-err');
  errEl.textContent='';
  try{
    const r=await fetch('/api/verify-password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
    const d=await r.json();
    if(d.ok){sessionStorage.setItem('bot_pw',pw);document.getElementById('pw-gate').style.display='none';return;}
  }catch(e){}
  errEl.textContent='Incorrect password. Please try again.';
  document.getElementById('pw-inp').value='';document.getElementById('pw-inp').focus();
}
checkBotPassword();
</script>
</body>
</html>"""


# ===================================================================
# ADMIN DASHBOARD (Item 20)
# ===================================================================

ADMIN_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Strat AI &#8212; Admin Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:#060b18;color:#e0e0e0;min-height:100vh}
#login-screen{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;gap:14px}
.logo-text{font-size:20px;font-weight:700;color:#5bb8f5;letter-spacing:-.5px;margin-bottom:4px}
#login-screen input{padding:11px 16px;background:#0f1830;border:1px solid #1a2a50;border-radius:8px;color:#e0e0e0;font-size:14px;width:280px;outline:none;font-family:inherit;transition:border .15s}
#login-screen input:focus{border-color:#5bb8f5}
.login-btn{padding:11px 0;background:linear-gradient(135deg,#0a2a5a,#1a6fb5);color:#fff;border:none;border-radius:8px;font-weight:600;cursor:pointer;font-size:14px;width:280px;transition:opacity .15s}
.login-btn:hover{opacity:.85}
.lerr{color:#f87171;font-size:13px;min-height:18px}
#dashboard{display:none;min-height:100vh}
.dash-hdr{display:flex;align-items:center;justify-content:space-between;padding:16px 26px;border-bottom:1px solid #1a2a50;background:#060b18;position:sticky;top:0;z-index:20}
.dash-hdr h1{font-size:16px;font-weight:700;color:#fff}
.hdr-sub{font-size:11px;color:#6688aa;margin-top:2px}
.hdr-right{display:flex;align-items:center;gap:10px}
.btn{padding:7px 15px;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;transition:opacity .15s}
.btn:hover{opacity:.8}
.btn-blue{background:#1a6fb5;color:#fff}
.btn-ghost{background:#0f1830;color:#8ba0c0;border:1px solid #1a2a50}
.lu{font-size:11px;color:#3a5070}
.body{padding:22px 26px}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:22px}
.kpi{background:#0f1830;border:1px solid #1a2a50;border-radius:11px;padding:16px;text-align:center;transition:border-color .2s}
.kpi:hover{border-color:#2a4a80}
.kn{font-size:26px;font-weight:700;line-height:1;margin-bottom:5px}
.kn.blue{color:#5bb8f5}.kn.green{color:#4ade80}.kn.yellow{color:#fbbf24}
.kl{font-size:10px;color:#6688aa;text-transform:uppercase;letter-spacing:.5px}
.ks{font-size:11px;color:#3a5070;margin-top:3px}
.charts-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:13px;margin-bottom:22px}
.chart-card{background:#0f1830;border:1px solid #1a2a50;border-radius:11px;padding:15px}
.ct{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:#6688aa;margin-bottom:11px}
.br{display:flex;align-items:center;gap:7px;margin-bottom:6px}
.bl{font-size:10px;color:#6688aa;width:72px;text-align:right;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bt{flex:1;background:#0a1428;border-radius:3px;height:15px;overflow:hidden}
.bf{height:100%;border-radius:3px;display:flex;align-items:center;padding-left:4px;transition:width .5s ease}
.bf span{font-size:9px;font-weight:700;color:rgba(255,255,255,.8);white-space:nowrap}
.bc{font-size:10px;color:#5bb8f5;width:26px;text-align:right;flex-shrink:0}
.filters{display:flex;align-items:center;flex-wrap:wrap;gap:9px;margin-bottom:13px}
.filters input[type=search]{padding:7px 12px;background:#0f1830;border:1px solid #1a2a50;border-radius:7px;color:#e0e0e0;font-size:13px;width:230px;outline:none;font-family:inherit;transition:border .15s}
.filters input[type=search]:focus{border-color:#5bb8f5}
.fg{display:flex;gap:4px;flex-wrap:wrap}
.fb{padding:5px 11px;border-radius:18px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;border:1px solid #1a2a50;background:#0a1428;color:#6688aa;transition:all .15s}
.fb.on{background:#1a3a6a;color:#5bb8f5;border-color:#2a5a9a}
.fb:hover:not(.on){background:#0f1e38;color:#aac}
.tw{overflow-x:auto;border-radius:11px;border:1px solid #1a2a50}
table{width:100%;border-collapse:collapse;background:#0f1830;min-width:960px}
th{background:#0a1e3d;color:#5bb8f5;font-size:10px;text-transform:uppercase;letter-spacing:.5px;padding:11px 13px;text-align:left;font-weight:700;white-space:nowrap}
td{padding:9px 13px;font-size:13px;border-top:1px solid #1a2540;color:#bbb;vertical-align:middle}
tr:hover td{background:#0a1e3d;cursor:pointer}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.3px}
.bg{background:#052e16;color:#4ade80}
.bm{background:#422006;color:#fbbf24}
.bp{background:#450a0a;color:#f87171}
.bu{background:#1a2540;color:#6688aa}
.bb{background:#0a2a50;color:#5bb8f5}
.bl2{background:#1a1050;color:#a78bfa}
.bhy{background:#0a2020;color:#34d399}
.ap{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:700;background:rgba(91,184,245,.12);color:#5bb8f5;border:1px solid rgba(91,184,245,.25)}
.hot{animation:hg 1.6s ease-in-out infinite alternate}
@keyframes hg{from{box-shadow:0 0 3px rgba(74,222,128,.2)}to{box-shadow:0 0 9px rgba(74,222,128,.4)}}
.empty{text-align:center;padding:36px;color:#334466;font-size:14px}
.trunc{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px;display:block}
#modal{position:fixed;inset:0;z-index:100;display:none}
#modal.open{display:flex;align-items:flex-start;justify-content:flex-end}
.mo{position:absolute;inset:0;background:rgba(0,0,0,.55);backdrop-filter:blur(2px)}
.mp{position:relative;z-index:1;background:#080e1e;border-left:1px solid #1a2a50;width:600px;max-width:97vw;height:100vh;overflow-y:auto;display:flex;flex-direction:column;animation:sldin .22s ease}
@keyframes sldin{from{transform:translateX(40px);opacity:0}to{transform:none;opacity:1}}
.mh{position:sticky;top:0;background:#080e1e;border-bottom:1px solid #1a2a50;padding:15px 20px;display:flex;align-items:flex-start;justify-content:space-between;gap:12px;z-index:2}
.mhi h2{font-size:15px;font-weight:700;color:#fff;margin-bottom:4px}
.mhi .mmeta{font-size:12px;color:#6688aa;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.mcl{background:none;border:1px solid #1a2a50;color:#8ba0c0;border-radius:6px;padding:3px 9px;cursor:pointer;font-size:15px;line-height:1;transition:all .15s;flex-shrink:0;margin-top:1px}
.mcl:hover{background:#1a2a50;color:#fff}
.mb{padding:18px 20px;flex:1}
.ms{margin-bottom:20px}
.mst{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:#5bb8f5;margin-bottom:9px;padding-bottom:5px;border-bottom:1px solid #1a2a50;display:flex;align-items:center;justify-content:space-between}
.ig{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.ii label{font-size:10px;text-transform:uppercase;color:#3a5070;letter-spacing:.4px;display:block;margin-bottom:2px}
.ii span{font-size:13px;color:#ccc}
.sy{background:#0a1428;border:1px solid #1a2a50;border-radius:7px;padding:12px;font-size:13px;line-height:1.65;color:#b8c4d8;white-space:pre-wrap;max-height:280px;overflow-y:auto}
.qa{margin-bottom:9px;padding:9px 11px;background:#0a1428;border-radius:6px;border-left:2px solid #1a4a8a}
.qq{font-size:10px;color:#5bb8f5;margin-bottom:3px;font-weight:700;text-transform:uppercase;letter-spacing:.3px}
.qa-a{font-size:13px;color:#b8c4d8;line-height:1.5}
.ftag{display:inline-block;padding:3px 10px;background:#1a2540;border-radius:11px;font-size:11px;color:#8ba0c0;margin:2px}
.rep{background:#050b1a;border:1px solid #1a3a6a;border-radius:7px;padding:14px;font-size:12px;line-height:1.75;color:#c8d4e8;white-space:pre-wrap;max-height:480px;overflow-y:auto;font-family:'Courier New',monospace}
.mf{position:sticky;bottom:0;background:#080e1e;border-top:1px solid #1a2a50;padding:12px 20px;display:flex;gap:8px;flex-wrap:wrap}
.cb{background:none;border:none;color:#5bb8f5;font-size:10px;cursor:pointer;text-decoration:underline;font-family:inherit;margin-left:6px}
@media(max-width:700px){.body{padding:14px}.kpi-grid{grid-template-columns:repeat(2,1fr)}.charts-row{grid-template-columns:1fr}.mp{width:100vw}}
</style>
</head>
<body>

<div id="login-screen">
  <div class="logo-text">Strat AI Solutions</div>
  <p style="font-size:13px;color:#6688aa;margin-bottom:6px">Admin Dashboard</p>
  <input type="password" id="pw" placeholder="Admin password" onkeydown="if(event.key==='Enter')login()">
  <button class="login-btn" onclick="login()">Login</button>
  <div class="lerr" id="lerr"></div>
</div>

<div id="dashboard">
  <div class="dash-hdr">
    <div>
      <h1>Strat AI Solutions &mdash; Admin Dashboard</h1>
      <div class="hdr-sub">Audit sessions, leads &amp; performance KPIs</div>
    </div>
    <div class="hdr-right">
      <span class="lu" id="lu"></span>
      <button class="btn btn-blue" onclick="loadData()">&#8635; Refresh</button>
      <button class="btn btn-ghost" onclick="logout()">Logout</button>
    </div>
  </div>

  <div class="body">
    <div class="kpi-grid" id="kpi-grid"></div>
    <div class="charts-row" id="charts-row"></div>

    <div class="filters">
      <input type="search" id="srch" placeholder="Search company, contact, email&hellip;" oninput="renderTable()">
      <div class="fg">
        <button class="fb on" data-v="" onclick="setF('fit',this)">All Fit</button>
        <button class="fb" data-v="good" onclick="setF('fit',this)">Good</button>
        <button class="fb" data-v="moderate" onclick="setF('fit',this)">Moderate</button>
        <button class="fb" data-v="poor" onclick="setF('fit',this)">Poor</button>
      </div>
      <div class="fg">
        <button class="fb on" data-v="" onclick="setF('type',this)">All Types</button>
        <button class="fb" data-v="broker" onclick="setF('type',this)">Broker</button>
        <button class="fb" data-v="lender" onclick="setF('type',this)">Lender</button>
        <button class="fb" data-v="hybrid" onclick="setF('type',this)">Hybrid</button>
      </div>
    </div>

    <div class="tw">
      <table>
        <thead>
          <tr>
            <th>Date</th><th>Company</th><th>Contact</th><th>Type</th><th>Fit</th>
            <th>Stage</th><th>Top Bottleneck</th><th>API Cost</th><th>Call Link</th><th>Audits</th><th>Actions</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<div id="modal">
  <div class="mo" onclick="closeModal()"></div>
  <div class="mp">
    <div class="mh" id="mh"></div>
    <div class="mb" id="mb"></div>
    <div class="mf" id="mf"></div>
  </div>
</div>

<script>
let token='',allSessions=[],activeF={fit:'',type:''};
const SL={'intake':'Getting Started','classify':'Classification','qualify':'Qualification','snapshot':'Snapshot Audit','synthesis':'Synthesis','deep_audit':'Deep Audit','proposal_ready':'Proposal Ready','complete':'Complete'};
const SO=['intake','classify','qualify','snapshot','synthesis','deep_audit','proposal_ready','complete'];
const TL={'broker':'Broker','lender':'Lender','hybrid':'Hybrid','unknown':'—'};
const FL={'good':'Good','moderate':'Moderate','poor':'Poor','unknown':'—'};
const TC={'broker':'bb','lender':'bl2','hybrid':'bhy','unknown':'bu'};
const FC={'good':'bg','moderate':'bm','poor':'bp','unknown':'bu'};

function esc(s){if(!s)return'';return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function fmtDate(s){if(!s)return'—';const d=new Date(s);if(isNaN(d))return String(s);return d.toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'})+' '+d.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'});}
function fmtShort(s){if(!s)return'—';const d=new Date(s);if(isNaN(d))return String(s);return d.toLocaleDateString(undefined,{month:'short',day:'numeric',year:'2-digit'});}

async function login(){
  const pw=document.getElementById('pw').value;
  try{
    const r=await fetch('/admin/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
    const d=await r.json();
    if(d.ok){token=d.token;document.getElementById('login-screen').style.display='none';document.getElementById('dashboard').style.display='block';loadData();}
    else document.getElementById('lerr').textContent='Incorrect password';
  }catch(e){document.getElementById('lerr').textContent='Connection error';}
}
function logout(){token='';allSessions=[];document.getElementById('dashboard').style.display='none';document.getElementById('login-screen').style.display='flex';document.getElementById('pw').value='';}

async function loadData(){
  try{
    const r=await fetch('/admin/api/sessions',{headers:{'Authorization':'Bearer '+token}});
    if(!r.ok){if(r.status===401)logout();return;}
    const d=await r.json();
    allSessions=(d.sessions||[]).sort((a,b)=>new Date(b.created_at||0)-new Date(a.created_at||0));
    document.getElementById('lu').textContent='Updated '+new Date().toLocaleTimeString();
    renderKPIs();renderCharts();renderTable();
  }catch(e){console.error(e);}
}

function renderKPIs(){
  const s=allSessions,n=s.length;
  const done=s.filter(x=>x.stage==='complete'||x.stage==='proposal_ready').length;
  const good=s.filter(x=>x.fit==='good').length;
  const mod=s.filter(x=>x.fit==='moderate').length;
  const cal=s.filter(x=>x.calendly_clicked).length;
  const cost=s.reduce((a,x)=>a+(x.api_cost_usd||0),0);
  const avg=n?cost/n:0;
  const cr=n?Math.round(done/n*100):0;
  const cv=n?Math.round(good/n*100):0;
  document.getElementById('kpi-grid').innerHTML=[
    {n:n,l:'Total Sessions',s:'All time',c:'blue'},
    {n:done,l:'Completed Audits',s:cr+'% completion rate',c:done?'green':'blue'},
    {n:good,l:'Good Fit Leads',s:cv+'% of sessions',c:good?'green':'blue'},
    {n:mod,l:'Moderate Fit',s:(n?Math.round(mod/n*100):0)+'% of sessions',c:mod?'yellow':'blue'},
    {n:cal,l:'Call Link Clicks',s:(n?Math.round(cal/n*100):0)+'% click rate',c:cal?'green':'blue'},
    {n:'$'+cost.toFixed(2),l:'Total API Cost',s:'$'+avg.toFixed(3)+' avg/session',c:'blue'},
  ].map(k=>`<div class="kpi"><div class="kn ${k.c}">${k.n}</div><div class="kl">${k.l}</div><div class="ks">${k.s}</div></div>`).join('');
}

function mkBar(label,count,total,color){
  const p=total?Math.round(count/total*100):0;
  return `<div class="br"><div class="bl">${label}</div><div class="bt"><div class="bf" style="width:${p}%;background:${color}"><span>${p}%</span></div></div><div class="bc">${count}</div></div>`;
}

function renderCharts(){
  const s=allSessions,n=s.length||1;
  const fc={good:0,moderate:0,poor:0,unknown:0};s.forEach(x=>{fc[x.fit||'unknown']=(fc[x.fit||'unknown']||0)+1;});
  const tc={broker:0,lender:0,hybrid:0,unknown:0};s.forEach(x=>{tc[x.client_type||'unknown']=(tc[x.client_type||'unknown']||0)+1;});
  const sf=['classify','qualify','snapshot','synthesis','deep_audit','proposal_ready','complete'];
  const sl2={'classify':'Classify','qualify':'Qualify','snapshot':'Snapshot','synthesis':'Synthesis','deep_audit':'Deep Audit','proposal_ready':'Proposal','complete':'Complete'};
  const sc={};sf.forEach(st=>{const idx=SO.indexOf(st);sc[st]=s.filter(x=>SO.indexOf(x.stage||'intake')>=idx).length;});
  const mx=Math.max(...Object.values(sc),1);
  document.getElementById('charts-row').innerHTML=
    `<div class="chart-card"><div class="ct">Fit Distribution</div>${mkBar('Good',fc.good,n,'#4ade80')}${mkBar('Moderate',fc.moderate,n,'#fbbf24')}${mkBar('Poor',fc.poor,n,'#f87171')}${mkBar('Unknown',fc.unknown,n,'#2a3a5a')}</div>`+
    `<div class="chart-card"><div class="ct">Client Type</div>${mkBar('Broker',tc.broker,n,'#5bb8f5')}${mkBar('Lender',tc.lender,n,'#a78bfa')}${mkBar('Hybrid',tc.hybrid,n,'#34d399')}${mkBar('Unknown',tc.unknown,n,'#2a3a5a')}</div>`+
    `<div class="chart-card"><div class="ct">Stage Funnel</div>${sf.map(st=>`<div class="br"><div class="bl" style="font-size:10px;width:68px">${sl2[st]}</div><div class="bt"><div class="bf" style="width:${Math.round(sc[st]/mx*100)}%;background:#1a6fb5"><span>${Math.round(sc[st]/(allSessions.length||1)*100)}%</span></div></div><div class="bc">${sc[st]}</div></div>`).join('')}</div>`;
}

function setF(key,btn){btn.closest('.fg').querySelectorAll('.fb').forEach(b=>b.classList.remove('on'));btn.classList.add('on');activeF[key]=btn.dataset.v;renderTable();}

function getFiltered(){
  const q=(document.getElementById('srch').value||'').toLowerCase().trim();
  return allSessions.filter(s=>{
    if(activeF.fit&&s.fit!==activeF.fit)return false;
    if(activeF.type&&s.client_type!==activeF.type)return false;
    if(q&&![s.company_name,s.contact_name,s.contact_email,s.id].join(' ').toLowerCase().includes(q))return false;
    return true;
  });
}

function renderTable(){
  const rows=getFiltered();
  const tb=document.getElementById('tbody');
  if(!rows.length){tb.innerHTML='<tr><td colspan="11" class="empty">No sessions match the current filters.</td></tr>';return;}
  tb.innerHTML=rows.map(s=>{
    const fit=s.fit||'unknown',type=s.client_type||'unknown',audits=s.user_audit_count||1;
    return `<tr class="${fit==='good'?'hot':''}" onclick="openModal('${esc(s.id)}')">
      <td style="font-size:12px;white-space:nowrap">${fmtShort(s.created_at)}</td>
      <td><span class="trunc" title="${esc(s.company_name)}">${esc(s.company_name||'—')}</span></td>
      <td>${esc(s.contact_name||'—')}<div style="font-size:11px;color:#4a6080">${esc(s.contact_email||'')}</div></td>
      <td><span class="badge ${TC[type]||'bu'}">${TL[type]||type}</span></td>
      <td><span class="badge ${FC[fit]||'bu'}">${FL[fit]||fit}</span></td>
      <td style="font-size:12px">${SL[s.stage]||s.stage||'—'}</td>
      <td><span class="trunc" style="font-size:12px" title="${esc(s.top_bottleneck)}">${esc(s.top_bottleneck||'—')}</span></td>
      <td style="font-size:12px">$${(s.api_cost_usd||0).toFixed(3)}</td>
      <td style="font-size:12px">${s.calendly_clicked?'<span style="color:#4ade80;font-weight:700" title="Call link was clicked (booking not confirmed)">✓</span>':'—'}</td>
      <td><span class="ap">${audits}</span></td>
      <td onclick="event.stopPropagation()"><button onclick="dlTx('${esc(s.id)}')" style="padding:3px 8px;background:#0a1e3d;color:#5bb8f5;border:1px solid #1a3a6a;border-radius:5px;font-size:11px;cursor:pointer">Transcript</button></td>
    </tr>`;
  }).join('');
}

async function openModal(sid){
  document.getElementById('modal').classList.add('open');
  document.getElementById('mh').innerHTML='<div class="mhi"><h2>Loading…</h2></div><button class="mcl" onclick="closeModal()">✕</button>';
  document.getElementById('mb').innerHTML='<div style="padding:30px;text-align:center;color:#334466">Loading session data…</div>';
  document.getElementById('mf').innerHTML='';
  try{
    const r=await fetch('/admin/api/session/'+sid,{headers:{'Authorization':'Bearer '+token}});
    if(!r.ok)throw new Error('HTTP '+r.status);
    renderModal(await r.json());
  }catch(e){document.getElementById('mb').innerHTML='<div style="padding:20px;color:#f87171">Failed to load: '+esc(e.message)+'</div>';}
}

function renderModal(s){
  const fit=s.fit||'unknown',type=s.client_type||'unknown';
  document.getElementById('mh').innerHTML=`
    <div class="mhi">
      <h2>${esc(s.company_name||'Unknown Company')}</h2>
      <div class="mmeta">
        ${esc(s.contact_name||'')}${s.contact_email?' · <span style="color:#4a6080">'+esc(s.contact_email)+'</span>':''}
        <span class="badge ${FC[fit]||'bu'}">${FL[fit]}</span>
        <span class="badge ${TC[type]||'bu'}">${TL[type]}</span>
      </div>
    </div>
    <button class="mcl" onclick="closeModal()">✕</button>`;

  let b='';
  b+=`<div class="ms"><div class="mst">Session Info</div><div class="ig">
    <div class="ii"><label>Started</label><span>${fmtDate(s.created_at)}</span></div>
    <div class="ii"><label>Completed</label><span>${s.completed_at?fmtDate(s.completed_at):'—'}</span></div>
    <div class="ii"><label>Stage</label><span>${SL[s.stage]||s.stage||'—'}</span></div>
    <div class="ii"><label>Fit</label><span><span class="badge ${FC[fit]||'bu'}">${FL[fit]}</span></span></div>
    <div class="ii"><label>API Calls</label><span>${s.api_calls||0}</span></div>
    <div class="ii"><label>API Cost</label><span>$${(s.api_cost_usd||0).toFixed(4)}</span></div>
    <div class="ii"><label>Call Link Clicked</label><span>${s.calendly_clicked?'Link Clicked (booking not confirmed)':'Not Clicked'}</span></div>
    <div class="ii"><label>Session ID</label><span style="font-size:10px;color:#334466">${esc(s.id||'')}</span></div>
  </div></div>`;

  if(s.flags&&s.flags.length){
    b+=`<div class="ms"><div class="mst">Flags</div><div>${s.flags.map(f=>`<span class="ftag">${esc(f)}</span>`).join('')}</div></div>`;
  }
  if(s.synthesis_text){
    b+=`<div class="ms"><div class="mst">Synthesis Summary</div><div class="sy">${esc(s.synthesis_text)}</div></div>`;
  }

  function qaSection(title,data){
    const keys=Object.keys(data||{});
    if(!keys.length)return'';
    const id='sc'+Math.random().toString(36).slice(2,8);
    return`<div class="ms"><div class="mst">${title} <button class="cb" onclick="togSec(this,'${id}')">hide</button></div><div id="${id}">${keys.map(k=>`<div class="qa"><div class="qq">${esc(k)}</div><div class="qa-a">${esc(String(data[k]))}</div></div>`).join('')}</div></div>`;
  }
  b+=qaSection('Qualification Answers',s.qual_data);
  b+=qaSection('Snapshot Answers',s.snapshot_answers);
  b+=qaSection('Deep Audit Answers',s.deep_answers);
  if(s.feedback&&s.feedback.rating){
    const stars='★'.repeat(s.feedback.rating)+'☆'.repeat(5-s.feedback.rating);
    b+=`<div class="ms"><div class="mst">User Feedback</div><div class="ig">
      <div class="ii"><label>Rating</label><span style="color:#fbbf24;font-size:15px">${stars} <span style="color:#ccc;font-size:12px">(${s.feedback.rating}/5)</span></span></div>
      <div class="ii"><label>Submitted</label><span>${s.feedback.submitted_at?fmtDate(s.feedback.submitted_at):'—'}</span></div>
      ${s.feedback.comment?`<div class="ii" style="grid-column:1/-1"><label>Comment</label><span>${esc(s.feedback.comment)}</span></div>`:''}
    </div></div>`;
  }
  b+=`<div class="ms" id="rep-sec" style="display:none"><div class="mst">Generated Audit Report</div><div class="rep" id="rep-box"></div></div>`;

  document.getElementById('mb').innerHTML=b;
  document.getElementById('mf').innerHTML=`
    <button class="btn btn-blue" onclick="genReport('${esc(s.id)}')">Generate Report</button>
    <button class="btn btn-ghost" onclick="dlTx('${esc(s.id)}')">Download Transcript</button>
    <button class="btn btn-ghost" onclick="closeModal()">Close</button>`;
}

function togSec(btn,id){const el=document.getElementById(id);if(!el)return;const h=el.style.display==='none';el.style.display=h?'':'none';btn.textContent=h?'hide':'show';}

async function genReport(sid){
  const sec=document.getElementById('rep-sec'),box=document.getElementById('rep-box');
  sec.style.display='block';box.textContent='Generating report… this may take 20-30 seconds.';
  sec.scrollIntoView({behavior:'smooth',block:'start'});
  try{
    const r=await fetch('/admin/api/session/'+sid+'/report',{headers:{'Authorization':'Bearer '+token}});
    if(!r.ok)throw new Error('HTTP '+r.status);
    const d=await r.json();
    box.textContent=d.report||'(No report generated)';
  }catch(e){box.textContent='Error generating report: '+e.message;}
}

async function dlTx(sid){
  try{
    const r=await fetch('/admin/api/session/'+sid+'/transcript',{headers:{'Authorization':'Bearer '+token}});
    if(!r.ok){alert('Transcript not available — session may still be in progress.');return;}
    const text=await r.text();
    const blob=new Blob([text],{type:'text/plain'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;a.download='transcript-'+sid.slice(0,8)+'.txt';a.click();
    URL.revokeObjectURL(url);
  }catch(e){alert('Error: '+e.message);}
}

function closeModal(){document.getElementById('modal').classList.remove('open');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal();});
</script>
</body>
</html>
"""


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
                    websocket=websocket,
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

            # Send streaming response with websocket
            result = await handle_message(session, user_text, websocket=websocket)
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


@app.post("/api/feedback/{session_id}")
async def submit_feedback(session_id: str, request: Request):
    session = store.get(session_id)
    body = await request.json()
    rating = int(body.get("rating", 0))
    comment = str(body.get("comment", ""))[:500]
    feedback_data = {"rating": rating, "comment": comment, "submitted_at": datetime.utcnow().isoformat()}
    if session:
        session.feedback = feedback_data
        asyncio.create_task(save_to_supabase(session))
    # Store in Supabase feedback table too if available
    return {"ok": True}


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


@app.post("/api/verify-password")
async def verify_bot_password(request: Request):
    if not BOT_PASSWORD:
        return JSONResponse({"ok": True})
    try:
        body = await request.json()
        if body.get("password") == BOT_PASSWORD:
            return JSONResponse({"ok": True})
    except Exception:
        pass
    return JSONResponse({"ok": False}, status_code=401)


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

    for s in sessions:
        if not s.get("top_bottleneck"):
            s["top_bottleneck"] = (s.get("metadata") or {}).get("top_bottleneck", "")
        if not s.get("top_opportunity"):
            s["top_opportunity"] = (s.get("metadata") or {}).get("top_opportunity", "")
        # If still empty, try flags
        if not s.get("top_bottleneck") and s.get("flags"):
            s["top_bottleneck"] = s["flags"][0].replace("_", " ").title()

    return {"sessions": sessions}


@app.get("/admin/api/session/{session_id}")
async def admin_session_detail(session_id: str, request: Request):
    _check_admin(request)
    session = store.get(session_id)
    if session:
        return session.to_dict()
    # Fall back to Supabase for sessions not currently in memory
    if _supabase_client:
        try:
            result = _supabase_client.table("sessions").select("*").eq("id", session_id).execute()
            if result.data:
                return result.data[0]
        except Exception as e:
            log.warning(f"Supabase detail fetch failed for {session_id}: {e}")
    raise HTTPException(404, "Session not found")


@app.get("/admin/api/session/{session_id}/transcript")
async def admin_session_transcript(session_id: str, request: Request):
    _check_admin(request)
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    lines = [f"STRAT AI AUDIT TRANSCRIPT\nSession: {session_id}\nClient: {session.contact_name} | {session.company_name}\nDate: {session.created_at}\n{'='*60}\n"]
    for msg in session.messages:
        role = "STRAT AI" if msg.get("role") == "bot" else (session.contact_name or "CLIENT").upper()
        ts = msg.get("ts", "")
        content = msg.get("content", "")
        lines.append(f"[{ts}] {role}:\n{content}\n")
    transcript = "\n---\n".join(lines)
    return Response(
        content=transcript,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="transcript-{session_id[:8]}.txt"'}
    )




@app.get("/admin/api/session/{session_id}/report")
async def admin_session_report(session_id: str, request: Request):
    _check_admin(request)
    session = store.get(session_id)
    if not session and _supabase_client:
        try:
            result = _supabase_client.table("sessions").select("*").eq("id", session_id).execute()
            if result.data:
                row = result.data[0]
                session = Session(
                    id=row.get("id", session_id),
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
        except Exception as e:
            log.warning(f"Supabase fetch for admin report {session_id}: {e}")
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
