---
permalink: /
title: "About"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hello, I'm Sibo Lu, a research developer(RD) with a passion for Recommendation and Search on Industry. This site serves as a portfolio of my work and a blog where I share insights on programming, technology, and personal projects.

Feel free to explore my projects, read my blog posts, and connect with me on social media. I'm always open to new opportunities and collaborations, so don't hesitate to reach out!

---

<h2 id="publications">Publications</h2>

{% if site.author.googlescholar %}
  <div class="wordwrap">You can also find my articles on <a href="{{site.author.googlescholar}}">my Google Scholar profile</a>.</div>
{% endif %}

{% include base_path %}

{% if site.publication_category %}
  {% for category in site.publication_category  %}
    {% assign title_shown = false %}
    {% for post in site.publications reversed %}
      {% if post.category != category[0] %}
        {% continue %}
      {% endif %}
      {% unless title_shown %}
<h3>{{ category[1].title }}</h3><hr />
        {% assign title_shown = true %}
      {% endunless %}
      {% include archive-single.html %}
    {% endfor %}
  {% endfor %}
{% else %}
  {% for post in site.publications reversed %}
    {% include archive-single.html %}
  {% endfor %}
{% endif %}

