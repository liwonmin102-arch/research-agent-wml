# research-agent-wml
research-agent project by wml
 #command to run
 cd research-agent
pip install -e .
uvicorn src.web:app --reload --port 8000

## Why I Built This

I'm a college student, and I do a lot of research across a wide range of topics — history, politics, technology news, and whatever broad, viral subject I'm trying to understand that week. My usual workflow looked like this:

1. Enter a topic keyword into Google.
2. Visit websites that look relevant.
3. Read each one.
4. Think about what I just read.
5. Form a summarized opinion in my head.
6. Repeat, while trying to keep track of any contradictions between sources.

That process works, but it's slow, and it's easy to lose track of where different sources agree, disagree, or push a particular slant. By the time I've read ten tabs, I can't always remember which claim came from where, and I definitely can't remember which sources had an industry agenda or a political lean.

I built Research Agent to automate that workflow end-to-end. I give it a topic, and a team of specialized AI agents handles the searching, reading, cross-referencing, and contradiction-tracking for me — then produces a structured report I can actually learn from, complete with a study guide and critical thinking questions.

It doesn't replace doing the research. It replaces the tedious parts so I can spend my time on the thinking part.