ROUTER_PROMPT = """Classify the user query into exactly one route.

Routes:
MSME: MSMEs, startups, entrepreneurship, business registration, Udyam registration, PMEGP, CGTMSE, Mudra loans, government schemes, subsidies, grants, funding, business loans, export businesses, import-export activities, taxation, compliance, incentives, and business support programs.
General: all non-business and non-MSME queries.

Return only MSME or General.
No explanations.

User query:
{query}
"""

SEARCH_QUERY_PROMPT = """Convert this user question into an optimized search query for finding MSME business advice in India covering government schemes, business registration, licenses, and funding options.
Output only the search query, maximum 20 words.

User question: {query}
"""

CONTEXT_SUFFICIENCY_PROMPT = """Question:
{query}

Retrieved Context:
{context}

Determine whether the retrieved context contains enough information to fully answer the question.

Respond ONLY with JSON:

{{"answerable": true}}

or

{{"answerable": false}}
"""

ANSWER_PROMPT = """You are an MSME and Startup Business Advisor for India.

Use the provided context to answer the user's question.

Knowledge Base:
{local_context}

Latest Information:
{live_context}

Question:
{question}

Instructions:
- Use only the information provided in the contexts.
- If the answer is not available in the contexts, say: "I don't have sufficient information to answer that."
- When discussing government schemes, include eligibility, benefits, and application process if available.
- When explaining how to start a business, include registrations, licenses, estimated costs, funding options, and relevant schemes if available.
- Use clear headings and bullet points.
- Be concise and factual.
- Do not make assumptions or invent information.
"""

NO_CONTEXT_ANSWER = 'I don\'t have sufficient information to answer that.'

OUT_OF_SCOPE_TEMPLATE = "{query}\nThe query does not fall within the scope of MSME, startup, or entrepreneurship-related subjects."

BLOCKED_TEMPLATE = "Your request cannot be processed because it violates the system's content validation rules.\n\nReason: {reason}"
