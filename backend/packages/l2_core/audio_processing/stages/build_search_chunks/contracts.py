from pydantic import BaseModel, Field


class TopicSection(BaseModel):
    start_utterance_index: int = Field(ge=0)
    end_utterance_index: int = Field(ge=0)
    topic: str


class TopicSectionsOutput(BaseModel):
    sections: list[TopicSection]
