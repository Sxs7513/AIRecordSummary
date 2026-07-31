from pydantic import BaseModel, Field


class TopicSection(BaseModel):
    start_utterance_index: int = Field(ge=0)
    end_utterance_index: int = Field(ge=0)
    topic: str
    terms: list[str] = Field(default_factory=list, max_length=8)
    search_context: str | None = Field(default=None, max_length=300)


class TopicSectionsOutput(BaseModel):
    sections: list[TopicSection]
