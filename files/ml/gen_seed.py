import random

# ── Helpers ──────────────────────────────────────────────────────────────────

senders = [
    "john", "sarah", "boss", "mom", "dad", "teacher", "professor", "hr",
    "amazon", "netflix", "google", "apple", "paypal", "bank", "school",
    "university", "client", "teammate", "manager", "recruiter", "doctor",
    "dentist", "landlord", "government", "irs", "shopee", "grab", "lazada",
    "spotify", "github", "linkedin", "twitter", "facebook", "insurance",
    "accountant", "lawyer", "supplier", "customer", "vendor", "noreply",
    "support", "admin", "info", "newsletter", "promotions", "sales", "billing",
]

subjects = [
    "invoice", "meeting", "assignment", "project", "deadline", "payment",
    "receipt", "newsletter", "promotion", "discount", "offer", "update",
    "notification", "reminder", "confirmation", "order", "shipping", "delivery",
    "application", "interview", "job offer", "contract", "report", "summary",
    "follow up", "feedback", "review", "subscription", "alert", "warning",
    "security", "password", "verification", "welcome", "onboarding", "event",
    "invitation", "schedule", "calendar", "homework", "exam", "grades",
]

email_refs = [
    "email {n}", "email number {n}", "number {n}", "#{n}", "the {n}st one",
    "the {n}nd one", "the {n}rd one", "message {n}", "mail {n}",
    "the latest email", "the last email", "the first email", "the top one",
    "the newest one", "the most recent one", "the oldest one",
    "that email", "that message", "that mail", "the email from {sender}",
    "the message from {sender}", "the one from {sender}",
    "the email about {subject}", "the message about {subject}",
    "the {subject} email", "the {subject} message",
    "that {subject} one", "the email at the top",
]

def ref():
    n = random.randint(1, 10)
    s = random.choice(senders)
    subj = random.choice(subjects)
    tmpl = random.choice(email_refs)
    return tmpl.format(n=n, sender=s, subject=subj)

def vary(templates):
    """Return all templates, randomly shuffled."""
    return list(templates)

# ── Intent templates ─────────────────────────────────────────────────────────

INTENTS = {

"fetch_inbox": [
    "show my inbox", "check my emails", "what emails do i have",
    "list my emails", "get my emails", "fetch inbox", "what's in my inbox",
    "whats in my inbox", "show me my emails", "new emails", "read my emails",
    "load inbox", "pull up inbox", "check inbox", "any new emails",
    "open my inbox", "inbox please", "inbox pls", "show inbox",
    "yo inbox", "anything new", "anything new in my mail", "check mail",
    "check mail pls", "new stuff", "new stuff in inbox", "got any emails",
    "do i have emails", "do i have any emails", "do i have new emails",
    "any messages", "any new messages", "show me my messages",
    "what messages do i have", "pull up my emails", "load my emails",
    "fetch my emails", "get inbox", "give me my emails", "emails please",
    "let me see my inbox", "let me see my emails", "show me inbox",
    "open inbox", "launch inbox", "display inbox", "display emails",
    "bring up inbox", "bring up my emails", "my inbox", "my emails",
    "emails", "inbox", "mail", "my mail", "check my mail", "show my mail",
    "what mail do i have", "got mail", "any mail", "new mail",
    "read mail", "open mail", "fetch mail", "get mail", "list mail",
    "pull mail", "load mail", "display mail", "bring up mail",
    "uh inbox", "inbox?", "emails?", "mail?", "anything?",
    "check", "check it", "check now", "quick check", "fast check",
    "what did i get", "what have i got", "whats new", "what's new",
    "show unread", "unread emails", "unread messages", "unread mail",
    "show unread emails", "get unread", "fetch unread",
    "show me unread", "list unread", "pull up unread",
    "recent emails", "recent mail", "recent messages",
    "show recent emails", "latest emails", "latest mail",
    "show latest", "get latest", "fetch latest",
    "can you check my inbox", "can you get my emails",
    "can you show me my emails", "could you check my mail",
    "please check inbox", "please show inbox", "please get emails",
    "i want to see my inbox", "i want to check my emails",
    "i need to see my emails", "i need to check inbox",
    "let me check my emails", "let me look at my inbox",
    "hey check my inbox", "hey show me my emails",
    "yo check inbox", "yo show emails", "yo get mail",
    "bro check inbox", "bro show emails",
    "what's waiting for me", "what's sitting in my inbox",
    "anything waiting", "anything sitting there",
    "give me an inbox update", "inbox update", "email update",
    "update me on my inbox", "what's going on in my inbox",
    "how many emails do i have", "how many unread do i have",
],

"fetch_priority": [
    "show urgent emails", "what's important", "priority inbox",
    "show me important emails", "sort by priority", "what needs my attention",
    "urgent emails", "rank my emails", "show emails by priority",
    "what are my most important emails", "critical emails", "high priority",
    "priority emails", "what should i deal with first",
    "what's most important", "whats most important",
    "show me what matters", "what matters most",
    "important stuff", "urgent stuff", "critical stuff",
    "what's pressing", "whats pressing", "pressing emails",
    "what needs attention", "needs attention", "attention needed",
    "show priority", "get priority emails", "fetch priority",
    "list priority", "priority list", "email priority",
    "rank emails", "sort emails by importance", "sort by importance",
    "most urgent", "most important emails", "top priority emails",
    "show me top priority", "show top priority", "get top priority",
    "what do i need to handle", "what do i need to deal with",
    "what should i respond to first", "what should i read first",
    "what's time sensitive", "time sensitive emails",
    "urgent messages", "important messages", "critical messages",
    "show important messages", "priority messages",
    "what's on fire", "anything on fire", "anything urgent",
    "anything critical", "anything important",
    "do i have anything urgent", "do i have anything important",
    "any urgent emails", "any important emails", "any critical emails",
    "high priority stuff", "high importance emails",
    "show me high priority", "show high priority",
    "what needs a quick response", "quick response needed",
    "what can't wait", "can't wait emails", "can't ignore",
    "what should i not ignore", "don't ignore these",
    "emails that matter", "emails i should care about",
    "sort inbox by priority", "order by priority",
    "order emails by importance", "ranked inbox",
    "give me the important ones", "just the important ones",
    "filter by priority", "filter important", "filter urgent",
],

"analyse": [
    "analyse email 1", "analyse email 2", "analyse email 3",
    "analyze email 1", "analyze 2", "summarise email 1",
    "summarize email 3", "what does email 2 say", "tell me about email 1",
    "read email 3", "open email 2", "analyse 4",
    "what's in email 1", "break down email 2", "details of email 3",
    f"analyse {ref()}", f"analyze {ref()}", f"summarise {ref()}",
    f"summarize {ref()}", f"open {ref()}", f"read {ref()}",
    f"what does {ref()} say", f"tell me about {ref()}",
    f"what's in {ref()}", f"break down {ref()}",
    f"details of {ref()}", f"give me details on {ref()}",
    f"show me {ref()}", f"expand {ref()}", f"elaborate on {ref()}",
    "open the latest email", "read the latest email",
    "what did the last email say", "show me that message",
    "what does that say", "what does it say", "what's it about",
    "read that", "read it", "open that", "open it",
    "show that email", "show the email", "show it",
    "what's that email about", "what's the email about",
    "give me the details", "more details", "more info",
    "tell me more", "expand on that", "elaborate",
    "summarise that", "summarize that", "summary please",
    "give me a summary", "quick summary", "brief summary",
    "what's happening in that email", "what did they say",
    "what did he say", "what did she say",
    "what did john say", "what did my boss say",
    "read the email from john", "read the one from sarah",
    "open the email from amazon", "show the message from hr",
    "what did amazon send", "what does netflix want",
    "what's the invoice say", "what's the meeting email say",
    "read the invoice", "open the meeting email",
    "read the assignment email", "what's the deadline email say",
    "check that email", "inspect that email", "look at that email",
    "look into that email", "go through that email",
    "decode that email", "translate that email",
    "what's going on in email 1", "what's happening in email 2",
    "give me the gist", "gist of email 1", "gist of that email",
    "what's the tldr", "tldr of email 1", "tldr please",
    "quick read of email 1", "quick look at email 2",
],

"archive": [
    "archive email 1", "archive email 2", "archive 3",
    "archive number 1", "move email 2 to archive",
    "put email 1 in archive", "get rid of email 3 from inbox",
    "remove email 1 from inbox", "file away email 2",
    f"archive {ref()}", f"move {ref()} to archive",
    f"put {ref()} in archive", f"file {ref()}",
    f"file away {ref()}", f"remove {ref()} from inbox",
    f"get {ref()} out of inbox", f"clear {ref()} from inbox",
    "archive that", "archive it", "archive this one",
    "archive the latest", "archive the last one",
    "archive the top one", "archive this email",
    "move to archive", "send to archive", "push to archive",
    "put in archive", "file it", "file that",
    "file this away", "file that away", "put it away",
    "clear from inbox", "remove from inbox", "get out of inbox",
    "get it out of my inbox", "get that out of my inbox",
    "clean up inbox", "tidy inbox", "declutter inbox",
    "archive all newsletters", "archive newsletters",
    "archive the newsletter", "archive promotions",
    "archive that promotion", "archive the promotional email",
    "archive the notification", "archive notifications",
    "stash that email", "stash it", "stash away email 1",
    "put email 3 away", "put it in the archive",
    "move that to archive", "move this to archive",
    "i don't need email 1 in inbox", "get email 2 out",
    "out of inbox email 1", "inbox zero email 2",
    "clear email 1", "clear it out", "clear that out",
    "sweep that away", "sweep email 1 away",
    "hide email 2", "hide that email",
],

"label": [
    "label email 1 as work", "label email 2 as urgent",
    "tag email 3 as important", "mark email 1 as personal",
    "categorise email 2 as school", "flag email 3 as review",
    "label 1 as finance", "tag 2 as follow up",
    "put email 3 under study",
    f"label {ref()} as work", f"label {ref()} as urgent",
    f"tag {ref()} as important", f"mark {ref()} as personal",
    f"categorise {ref()} as school", f"flag {ref()} as review",
    f"put {ref()} under finance", f"tag {ref()} as follow up",
    f"label {ref()} as finance", f"mark {ref()} as work",
    f"categorize {ref()} as urgent", f"flag {ref()} as important",
    "label that as work", "label it as urgent",
    "tag that as important", "mark that as personal",
    "flag that as review", "categorise that as school",
    "label this as finance", "tag this as follow up",
    "mark this email as work", "label the latest as urgent",
    "tag the last one as important", "mark the top one as personal",
    "put a label on email 1", "add label to email 2",
    "put work label on email 3", "add urgent tag to email 1",
    "label email 1", "tag email 2", "mark email 3",
    "categorise email 1", "flag email 2",
    "add tag to email 3", "apply label to email 1",
    "give email 2 a label", "give it the work label",
    "give that the finance tag", "mark it finance",
    "mark it urgent", "mark it personal", "mark it school",
    "tag it work", "tag it urgent", "tag it important",
    "label it work", "label it urgent", "label it personal",
    "flag it", "flag it as urgent", "flag it as important",
    "categorize it", "categorize it as work",
    "put it in the work category", "put under work",
    "assign work label", "assign urgent tag",
    "assign finance category", "assign it a label",
],

"trash": [
    "delete email 1", "trash email 2", "delete 3",
    "trash 1", "remove email 2", "get rid of email 1",
    "throw away email 3", "permanently delete email 2",
    f"delete {ref()}", f"trash {ref()}", f"remove {ref()}",
    f"get rid of {ref()}", f"throw away {ref()}",
    f"permanently delete {ref()}", f"bin {ref()}",
    f"dump {ref()}", f"discard {ref()}", f"nuke {ref()}",
    f"kill {ref()}", f"wipe {ref()}",
    "delete that", "delete it", "delete this one",
    "trash that", "trash it", "trash this",
    "remove that", "remove it", "remove this",
    "throw that away", "throw it away", "throw this away",
    "bin that", "bin it", "bin this",
    "dump that", "dump it", "dump this",
    "discard that", "discard it", "discard this",
    "get rid of that", "get rid of it", "get rid of this",
    "delete the latest", "trash the last one",
    "delete the top one", "trash the newest one",
    "permanently delete that", "permanently remove that",
    "nuke that email", "nuke it", "just delete it",
    "just trash it", "just remove it", "just bin it",
    "kill that email", "wipe that email", "destroy that email",
    "delete the email from amazon", "trash the newsletter",
    "delete the spam", "trash the spam", "delete spam",
    "trash spam", "bin the spam", "remove the spam",
    "delete junk", "trash junk", "bin junk",
    "delete that junk email", "trash that junk",
    "i don't want email 1", "don't need email 2",
    "get email 3 out of here", "lose email 1",
    "lose that email", "lose it",
],

"reply": [
    "reply to email 1", "draft a reply to email 2",
    "write a reply to email 3", "reply casually to email 1",
    "draft casual reply to email 2", "write professional reply to email 1",
    "respond to email 3", "send a reply to email 2",
    "reply formally to email 1", "compose reply to email 3",
    f"reply to {ref()}", f"draft a reply to {ref()}",
    f"write a reply to {ref()}", f"respond to {ref()}",
    f"send a reply to {ref()}", f"compose reply to {ref()}",
    f"reply casually to {ref()}", f"draft casual reply to {ref()}",
    f"write professional reply to {ref()}", f"reply formally to {ref()}",
    f"answer {ref()}", f"get back to {ref()}",
    f"write back to {ref()}", f"respond back to {ref()}",
    "reply to that", "reply to it", "reply to this",
    "draft a reply", "write a reply", "compose a reply",
    "respond to that", "respond to it", "respond to this",
    "send a reply", "send reply", "reply now",
    "reply casually", "casual reply", "keep it casual",
    "reply formally", "formal reply", "keep it formal",
    "professional reply", "write professional reply",
    "draft professional response", "formal response",
    "casual response", "write casual response",
    "reply to the latest", "reply to the last one",
    "respond to the latest email", "get back to the last email",
    "write back", "write back to that", "write back to it",
    "answer that", "answer it", "answer this email",
    "get back to that", "get back to it",
    "shoot a reply", "shoot back a reply", "fire back a reply",
    "quick reply to email 1", "quick response to email 2",
    "short reply to email 1", "brief reply to email 2",
    "send a quick reply", "fire off a reply",
    "compose response to email 1", "draft response to email 2",
    "write response to email 3", "respond with a reply",
],

"create_rule": [
    "archive all newsletters", "archive anything about promotions",
    "automatically archive marketing emails",
    "archive emails about google classroom",
    "label emails from boss as important",
    "mark emails from school as education",
    "archive all promotional emails",
    "create a rule to archive newsletters",
    "set up rule to label invoices as finance",
    "automatically label emails from hr",
    "whenever i get emails about sales archive them",
    "archive emails containing unsubscribe",
    "move all notification emails to archive",
    "label all emails from school",
    "trash all spam emails automatically",
    "always archive emails from noreply",
    "auto archive anything with discount in subject",
    "make a rule", "create a rule", "set up a rule",
    "add a rule", "new rule", "create automation",
    "set up automation", "add automation", "new automation",
    "make an automation", "automate this",
    f"always archive emails from {random.choice(senders)}",
    f"always label emails from {random.choice(senders)} as work",
    f"always trash emails from {random.choice(senders)}",
    f"automatically archive {random.choice(subjects)} emails",
    f"automatically label {random.choice(subjects)} emails as work",
    f"automatically trash {random.choice(subjects)} emails",
    f"archive everything from {random.choice(senders)}",
    f"label everything from {random.choice(senders)} as urgent",
    f"trash everything from {random.choice(senders)}",
    f"archive all {random.choice(subjects)} emails",
    f"label all {random.choice(subjects)} emails",
    f"trash all {random.choice(subjects)} emails",
    "whenever i get a newsletter archive it",
    "whenever i get a promotion trash it",
    "whenever i get spam delete it",
    "whenever there's a notification archive it",
    "if it's from noreply archive it",
    "if it's a newsletter trash it",
    "if it's spam delete it automatically",
    "if it contains unsubscribe archive it",
    "if it has discount in subject trash it",
    "if it's from amazon label it as shopping",
    "if it's from school label it as education",
    "if it's from boss label it as urgent",
    "set it so newsletters get archived",
    "set it so spam gets deleted",
    "set it so promotions get trashed",
    "rule: archive newsletters", "rule: trash spam",
    "rule: label boss emails as urgent",
    "auto handle newsletters", "auto handle spam",
    "auto handle promotions", "handle newsletters automatically",
    "deal with spam automatically", "deal with newsletters automatically",
],

"delete_rule": [
    "delete rule 1", "remove rule 2", "disable rule 3",
    "turn off rule 1", "stop rule 2", "cancel rule 3",
    "delete automation 1", "remove automation 2",
    "disable automation 3", "turn off automation 1",
    "stop automation 2", "cancel automation 1",
    f"delete rule {random.randint(1,10)}", f"remove rule {random.randint(1,10)}",
    f"disable rule {random.randint(1,10)}", f"turn off rule {random.randint(1,10)}",
    f"stop rule {random.randint(1,10)}", f"cancel rule {random.randint(1,10)}",
    "delete that rule", "remove that rule", "disable that rule",
    "turn off that rule", "stop that rule", "cancel that rule",
    "delete this rule", "remove this rule", "disable this rule",
    "turn off this rule", "stop this rule", "cancel this rule",
    "delete the rule", "remove the rule", "disable the rule",
    "turn off the rule", "stop the rule", "cancel the rule",
    "get rid of rule 1", "get rid of that rule",
    "kill rule 1", "kill that rule", "kill this rule",
    "nuke rule 1", "nuke that rule", "wipe rule 1",
    "clear rule 1", "clear that rule", "remove automation",
    "delete automation", "disable automation", "turn off automation",
    "stop automation", "cancel automation", "kill automation",
    "i don't want rule 1 anymore", "don't need rule 2",
    "rule 1 is not needed", "rule 2 can go",
    "get rid of automation 1", "lose rule 1",
    "deactivate rule 1", "deactivate that rule",
    "switch off rule 1", "switch off that rule",
    "pause rule 1", "pause that rule", "suspend rule 1",
],

"list_rules": [
    "show my rules", "list my rules", "what rules do i have",
    "show automation rules", "my automations", "view rules",
    "what automations are active", "show me my automations",
    "what have i set up",
    "list rules", "get rules", "fetch rules", "display rules",
    "show rules", "see rules", "check rules", "view automations",
    "list automations", "get automations", "fetch automations",
    "display automations", "see automations", "check automations",
    "what rules are running", "what automations are running",
    "active rules", "active automations", "running rules",
    "show active rules", "show running rules",
    "what rules exist", "what automations exist",
    "do i have any rules", "do i have any automations",
    "any rules set up", "any automations set up",
    "rules please", "automations please", "show me rules",
    "pull up rules", "pull up automations", "bring up rules",
    "what did i set up", "what have i automated",
    "what's automated", "what's been set up",
    "show me what's automated", "tell me my rules",
    "give me the rules", "give me my automations",
    "list everything i've set up", "what's running",
    "rules list", "automation list", "rule list",
    "my rule list", "my automation list",
    "how many rules do i have", "how many automations",
],

"list_history": [
    "show history", "what have you done", "show action history",
    "what actions did you take", "recent actions",
    "show me what you've done", "history", "audit log",
    "what did aatas do",
    "show log", "view log", "get log", "fetch log",
    "display log", "list log", "check log", "see log",
    "action log", "event log", "activity log",
    "show action log", "view action log", "get action log",
    "show event log", "show activity log",
    "what happened", "what's happened", "what's been happening",
    "what have you been doing", "what did you do",
    "what actions were taken", "what was done",
    "show me the log", "show me history", "show me the history",
    "give me the history", "give me the log",
    "pull up history", "pull up the log", "bring up history",
    "recent activity", "show recent activity", "view recent activity",
    "past actions", "show past actions", "list past actions",
    "previous actions", "show previous actions",
    "what did aatas do recently", "aatas history",
    "bot history", "bot log", "bot activity",
    "show bot history", "show bot log", "show bot activity",
    "log please", "history please", "activity please",
    "audit trail", "show audit trail", "view audit trail",
    "what's the audit trail", "action trail",
    "show me what happened", "tell me what happened",
    "run me through what happened", "recap",
    "give me a recap", "quick recap", "action recap",
],

"none": [
    "hello", "hi", "hey", "thanks", "ok", "good morning", "how are you",
    "what can you do", "help me", "what are you", "who are you",
    "cool", "nice", "got it", "alright",
    "lol", "bruh", "what", "idk", "that's crazy", "nah",
    "ok thanks bro", "lmao", "haha", "hehe", "wow",
    "omg", "wtf", "damn", "dang", "nice one",
    "ok cool", "sounds good", "got it thanks", "cheers",
    "thank you", "thanks a lot", "much appreciated", "appreciate it",
    "perfect", "great", "awesome", "sweet", "dope", "sick",
    "no worries", "no problem", "np", "np bro", "all good",
    "yeah", "yep", "yup", "sure", "of course", "definitely",
    "no", "nope", "nah bro", "not really", "i don't think so",
    "maybe", "perhaps", "possibly", "not sure", "idk man",
    "good", "bad", "okay", "fine", "whatever",
    "interesting", "fascinating", "cool beans", "neat",
    "what's up", "sup", "wassup", "yo", "hey there",
    "good afternoon", "good evening", "good night",
    "how's it going", "how are things", "all good?",
    "bye", "goodbye", "see ya", "later", "cya", "peace",
    "brb", "afk", "gtg", "ttyl", "talk later",
    "hmm", "uh", "um", "err", "ahh", "ohh",
    "wait what", "huh", "what do you mean", "i'm confused",
    "can you repeat that", "say that again", "what was that",
    "start over", "never mind", "forget it", "cancel",
    "stop", "quit", "exit", "done", "finished",
    "test", "testing", "hello world", "ping", "pong",
    "1", "2", "3", "abc", "xyz", "random text here",
    "i love you", "you're great", "you're awesome",
    "good bot", "bad bot", "smart bot", "dumb bot",
    "are you real", "are you human", "are you ai",
    "what time is it", "what day is it", "what's the date",
    "how's the weather", "is it raining", "what's the temp",
    "tell me a joke", "say something funny", "make me laugh",
    "i'm bored", "entertain me", "do something",
    "what's 2 plus 2", "what's 1 plus 1", "do math",
    "translate this", "speak french", "say hello in spanish",
    "ok i'm back", "i'm here", "still here", "still there",
    "hold on", "wait a sec", "one moment", "just a sec",
    "ok ready", "ready", "let's go", "go", "begin", "start",
],

}

# ── Generate 10,000 examples ──────────────────────────────────────────────────

random.seed(42)
all_examples = []

# How many per intent to hit ~10k
targets = {
    "fetch_inbox":    1000,
    "fetch_priority": 800,
    "analyse":        900,
    "archive":        900,
    "label":          900,
    "trash":          900,
    "reply":          900,
    "create_rule":    900,
    "delete_rule":    800,
    "list_rules":     800,
    "list_history":   800,
    "none":           1000,
}

fillers = [
    "please", "pls", "now", "quickly", "asap", "for me",
    "can you", "could you", "would you", "hey", "yo", "bro",
    "mate", "dude", "pal", "man", "sir",
    "", "", "", "",  # weighted toward no filler
]

def augment(text):
    """Randomly vary casing, punctuation, fillers."""
    variations = [text]
    # add filler prefix
    filler = random.choice(fillers)
    if filler:
        variations.append(f"{filler} {text}")
    # add filler suffix
    filler2 = random.choice(["please", "pls", "now", "asap", "thanks", "thx", ""])
    if filler2:
        variations.append(f"{text} {filler2}")
    # uppercase first letter
    variations.append(text.capitalize())
    # all caps (simulate shouting)
    if random.random() < 0.05:
        variations.append(text.upper())
    # trailing punctuation
    for p in ["?", "!", ".", "!!", "??", "..."]:
        if random.random() < 0.15:
            variations.append(text + p)
    return variations

for intent, target in targets.items():
    base = INTENTS[intent]
    pool = []
    # Expand with augmentations
    for phrase in base:
        pool.extend(augment(phrase))
    # Deduplicate
    pool = list(set(pool))
    # Sample up to target, with replacement if needed
    if len(pool) >= target:
        chosen = random.sample(pool, target)
    else:
        chosen = pool + random.choices(pool, k=target - len(pool))
    for text in chosen:
        all_examples.append((text.lower().strip(), intent))

# Shuffle
random.shuffle(all_examples)

print(f"Total examples: {len(all_examples)}")
# Count per intent
from collections import Counter
c = Counter(label for _, label in all_examples)
for intent, count in sorted(c.items()):
    print(f"  {intent}: {count}")

# Write seed_data.py
lines = ['"""\nAATAS — Expanded Seed Training Data\n10,000 examples across all intents.\nGenerated with varied phrasing, casual language, slang, and non-numeric email references.\n"""\n\nSEED_DATA = [\n']
for text, intent in all_examples:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'    ("{escaped}", "{intent}"),\n')
lines.append(']\n')

with open('/Users/erianeetiekhong/Documents/AATAS_PROJECT/files/ml/seed_data.py', 'w') as f:
    f.writelines(lines)

print("Written to /Users/erianeetiekhong/Documents/AATAS_PROJECT/files/ml/seed_data.py")
