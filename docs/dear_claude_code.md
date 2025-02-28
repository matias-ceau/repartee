- embedding module in config (default to openai v3 small)
- llm data location configurable, defaults to $XDG_DATA_HOME/.repartee
+ knowledge markdown folders configurable as well (i use obsidian but it doesnt have to rely on any of its features). however i store metadata in 'yaml' format at the start of the md files. The most imprtant metadata is `ntype`, with example types such as : album, person, artist (child of person class/type), atlas (sort of index of specific domain), journal, event, howto. etc. Your categories are great, but should not break with files that don't follow your specs
+ don't hesitate  to leverage jsonld or other rdf fileformat
+ use html dynamic graph representations like vis.js with physics or other lightweight and fast viewer tools
- roadmap and update to the README after all the changes and having asked key questions to me 
