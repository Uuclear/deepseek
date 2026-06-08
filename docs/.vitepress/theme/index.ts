import DefaultTheme from 'vitepress/theme'
import 'katex/dist/katex.min.css'
import './custom.css'
import LearningMeta from './components/LearningMeta.vue'
import ExampleBox from './components/ExampleBox.vue'
import HomeRoadmap from './components/HomeRoadmap.vue'
import Layout from './Layout.vue'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('LearningMeta', LearningMeta)
    app.component('ExampleBox', ExampleBox)
    app.component('HomeRoadmap', HomeRoadmap)
  },
}
