import siteMetadata from '@/data/siteMetadata'

const formatDate = (date, locale) => {
  const options = {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  }
  const now = new Date(date).toLocaleDateString(locale, options)

  return now
}

export default formatDate
