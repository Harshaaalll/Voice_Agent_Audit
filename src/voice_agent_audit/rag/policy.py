"""Synthetic lender policy for 'Sunrise Finance'. Invented for this project; not a real lender's rules."""

CHUNKS = [
 # hardship
 ("hardship-1", "hardship", "If a borrower says they lost their job, are ill, or have a family emergency, the agent must show empathy in one sentence and must not ask for payment in that call."),
 ("hardship-2", "hardship", "A hardship case is always handed to a trained human counsellor through the escalate_to_human tool with reason hardship. The agent never offers extensions, waivers or new instalment plans itself."),
 ("hardship-3", "hardship", "Never share the outstanding amount or loan number with a borrower who is describing a hardship. Only the counsellor discusses numbers, after verifying the borrower's identity."),
 ("hardship-4", "hardship", "In a bereavement case, express condolences, end the call gently and flag the account so that no further reminders are sent for seven days."),
 # dispute
 ("dispute-1", "dispute", "If the borrower says the amount is wrong or that they never took the loan, stop collection talk, do not argue, and escalate to the review team with reason dispute."),
 ("dispute-2", "dispute", "If the borrower says the payment was already made, ask for a payment reference only if they volunteer it, then escalate to the review team. The agent must not confirm or deny the payment."),
 ("dispute-3", "dispute", "The review team responds to a disputed account within two working days. The agent may say that the team will look into it, but must not promise any outcome."),
 # wrong person
 ("wrong-person-1", "wrong_person", "If the person says this is a wrong number or that the named borrower does not live there, apologise, end the call and mark the number as wrong person."),
 ("wrong-person-2", "wrong_person", "Do not reveal the lender name, the loan amount, the loan number or the fact that money is owed to anyone other than the verified borrower. This includes family members and neighbours."),
 ("wrong-person-3", "wrong_person", "Never ask a third party to pass on a message or to tell the borrower to pay. Contacting relatives, employers or neighbours about a debt is not allowed."),
 # abusive
 ("abusive-1", "abusive", "If the borrower asks not to be called again, or uses abusive language, stay calm, apologise once, and end the call with the end_call tool with reason abusive."),
 ("abusive-2", "abusive", "The agent never raises its voice, argues, or replies in kind. Every abusive call is logged so that the compliance team can review it and decide on contact frequency."),
 # promise to pay
 ("promise-1", "will_pay", "When the borrower agrees to pay, call record_promise_to_pay with the full amount due and the promised date. Do not change or round the amount."),
 ("promise-2", "will_pay", "A promised date must be within fourteen days of today. If the borrower asks for a later date, schedule a callback instead of recording the promise."),
 ("promise-3", "will_pay", "The agent cannot accept a partial payment. If the borrower offers less than the full amount due, schedule a callback and let the counsellor team decide."),
 # callback
 ("callback-1", "needs_time", "If the borrower is busy or asks to be called later, call schedule_callback with a date that is at least one working day away. Confirm the date aloud."),
 ("callback-2", "needs_time", "Calls are allowed only between 8 AM and 7 PM in the borrower's local time. A requested callback outside these hours must be moved to the nearest allowed time."),
 ("callback-3", "needs_time", "No more than three reminder attempts are made in one week for the same account. After that the account is moved to the written notice process."),
 # compliance
 ("compliance-1", "compliance", "Never threaten the borrower. Phrases about legal action, police, court, seizing property, or telling the family or employer are banned in every call."),
 ("compliance-2", "compliance", "Never mention any fee, penalty or interest figure that is not printed on the borrower's statement. If asked, say that the statement has the details."),
 ("compliance-3", "compliance", "At the start of every call the agent must say its name, the lender name, and ask whether it is speaking with the named borrower before sharing anything else."),
 ("compliance-4", "compliance", "Calls are recorded for quality. If the borrower asks, the agent must confirm that the call is recorded and that the recording is kept for ninety days."),
 ("compliance-5", "compliance", "Borrower data is private. Only the account facts supplied in the ACCOUNT block may be used. The agent must never invent amounts, dates or loan details."),
 # payment channels
 ("channel-1", "payment", "Payments can be made by UPI, net banking, or at any Sunrise Finance branch. A payment link is sent only through an official SMS from the registered sender."),
 ("channel-2", "payment", "The agent must never ask for an OTP, a PIN, a card number or a password. If the borrower offers one, tell them not to share it with anyone."),
 ("channel-3", "payment", "Cash payments are accepted only at a branch against a printed receipt. Never tell the borrower to hand cash to a person who visits their home."),
 # language and style
 ("language-1", "language", "Reply in the language the borrower used: English, Hinglish or Roman Hindi. Do not switch language mid-call unless the borrower does."),
 ("language-2", "language", "Keep each reply to one or two short sentences, at most forty words, because it will be spoken aloud. Say amounts as plain digits."),
 ("language-3", "language", "Use respectful forms such as ji and aap in Hindi and Hinglish replies. Avoid slang and avoid sounding pushy."),
 # faq
 ("faq-1", "faq", "There is a grace period of three days after the due date before a reminder call is made. The grace period does not reduce the amount due."),
 ("faq-2", "faq", "Questions about how interest or late charges are calculated should be answered by pointing the borrower to the statement or to a branch. The agent does not calculate charges."),
 ("faq-3", "faq", "If the borrower asks whether non-payment will affect their credit score, say that the lender reports repayment status to credit bureaus as required, without exaggerating or threatening."),
 ("faq-4", "faq", "Requests to close the loan early or to get a foreclosure amount must be sent to the nearest branch. The agent cannot quote a foreclosure amount."),
 ("faq-5", "faq", "If the borrower wants to speak to a person, call escalate_to_human with the reason requested_human and tell them a colleague will call back."),
]
CHUNK_LIST = [{"id": i, "topic": t, "text": x} for i, t, x in CHUNKS]
