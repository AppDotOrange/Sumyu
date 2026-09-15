#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ParamRange {
    pub start: usize,
    pub len: usize,
}

impl ParamRange {
    #[inline(always)]
    pub fn end(self) -> usize {
        self.start + self.len
    }
}

#[derive(Default, Clone)]
pub struct ParameterStore {
    pub values: Vec<f32>,
    pub grads: Vec<f32>,
}

impl ParameterStore {
    #[inline]
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            grads: Vec::new(),
        }
    }

    #[inline]
    pub fn alloc(&mut self, value: f32) -> usize {
        let index = self.values.len();

        self.values.push(value);
        self.grads.push(0.0);

        index
    }

    pub fn alloc_many<I>(&mut self, values: I) -> ParamRange
    where
        I: IntoIterator<Item = f32>,
    {
        let start = self.values.len();

        self.values.extend(values);

        let len = self.values.len() - start;

        self.grads.resize(self.values.len(), 0.0);

        ParamRange { start, len }
    }

    #[inline(always)]
    pub fn values(&self, range: ParamRange) -> &[f32] {
        &self.values[range.start..range.end()]
    }

    #[inline(always)]
    pub fn values_mut(&mut self, range: ParamRange) -> &mut [f32] {
        &mut self.values[range.start..range.end()]
    }

    #[inline(always)]
    pub fn grads(&self, range: ParamRange) -> &[f32] {
        &self.grads[range.start..range.end()]
    }

    #[inline(always)]
    pub fn grads_mut(&mut self, range: ParamRange) -> &mut [f32] {
        &mut self.grads[range.start..range.end()]
    }

    #[inline(always)]
    pub fn zero_grads(&mut self) {
        self.grads.fill(0.0);
    }

    #[inline(always)]
    pub fn parameter_count(&self) -> usize {
        self.values.len()
    }
}
