---
permalink: /
layout: homepage
title: ""
excerpt: ""
author_profile: false
sidebar_intro: "Research developer interested in recommendation, search, debiasing, and representation learning for recommender systems."
redirect_from: 
  - /about/
  - /about.html
---

<div class="anchor" id="about-me"></div>

I am a research developer with a passion for recommendation and search in industry. My recent work focuses on generative recommendation, agentic models, and multi-modal large foundation models.

This homepage gives a concise overview of my research interests and recent publications. You can browse selected papers below, or visit [Google Scholar]({{ site.author.googlescholar }}) for a fuller publication record.

<div class="anchor" id="research"></div>

# 🔬 Research Interests

- Recommendation systems and search
  - Generative Recommendation
- Large Foundation Model
  - Agentic Model
  - Multi-Modal

<div class="anchor" id="selected-publications"></div>

# 📝 Selected Publications

{% assign selected_publication_permalinks = "/publication/2026-era-meta-representation,/publication/2025-contrastive-intent-vae,/publication/2025-disentangled-multi-graph,/publication/2024-contrastive-disentangled" | split: "," %}
{% for publication_permalink in selected_publication_permalinks %}
  {% assign post = site.publications | where: "permalink", publication_permalink | first %}
  {% if post %}
    {% include publication-card.html post=post %}
  {% endif %}
{% endfor %}

<div class="anchor" id="all-publications"></div>

# 📚 All Publications

<ol class="pub-list compact-publication-list">
{% assign all_publications = site.publications | sort: "date" | reverse %}
{% for post in all_publications %}
  {% include archive-single.html %}
{% endfor %}
</ol>

<div class="anchor" id="contact"></div>

# ✉️ Contact

{% if site.author.employer %}
- **Affiliation:** {{ site.author.employer }}
{% endif %}
{% if site.author.location %}
- **Location:** {{ site.author.location }}
{% endif %}
{% if site.author.googlescholar %}
- **Google Scholar:** [Profile]({{ site.author.googlescholar }})
{% endif %}
{% if site.author.github %}
- **GitHub:** [{{ site.author.github }}](https://github.com/{{ site.author.github }})
{% endif %}
{% if site.author.email %}
- **Email:** [{{ site.author.email }}](mailto:{{ site.author.email }})
{% endif %}
